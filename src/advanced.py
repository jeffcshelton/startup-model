from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

MODEL_NAME = "advanced"
Q_MIN = 0.001 # optional: prevents virality process from becoming negative (more users decreasing product attractiveness)

# params are still measured annually
DEFAULT_PARAMS: dict[str, float | int] = {
    "p": 0.03, # coeffieicnt of innovation
    "q": 0.38, # coefficient of immitation
    "kappa": 1.0, # latent virality mean reversion speed (higher means latent q reverts to mean faster)
    "sigma_q": 0.05, # latent virality noise volatility (higher means latent q is more unpredictable)
    "K": 50_000.0, # market size
    "v": 100.0, # max revenue per customer
    "epsilon": 0.1, # price compression factor (higher means revenue per customer decreases faster as customers approach market size)
    "chi": 0.02, # customer churn rate
    "b0": 50_000.0, # fixed burn
    "gamma": 40.0, # cost per customer
    "alpha": 200.0, # customer acquisition cost
    "sigma_N": 5.0, # noise volatility for customer growth
    "N0": 10.0, # initial customers
    "C0": 2_000_000.0, # initial cash
    "T": 60, # number of time steps to simulate
    "dt": 1.0 / 12.0, # time step size (1/12 means monthly steps)
}

# central random number generator
def rng_from_seed(seed: int | None) -> np.random.Generator:
    """Build a random number generator."""

    return np.random.default_rng(seed)

# Bass diffusion customer gain (separated from churn to record customers gained/lost separately)
def acquisition_flow(customers: float, p: float, q: float, market_size: float) -> float:
    """Evaluate gross customer acquisition before churn."""

    return float((p + q * customers / market_size) * (market_size - customers))

# net customer growth drift (acquisition minus churn)
def customer_drift(customers: float, p: float, q: float, market_size: float, chi: float) -> float:
    """Evaluate deterministic customer drift."""

    return float(acquisition_flow(customers, p, q, market_size) - chi * customers)

# revenue per customer now a function of customer count to capture price compression effects as market saturates
def revenue(customers: float, market_size: float, v: float, epsilon: float) -> float:
    """Evaluate price-compressed revenue: v * (1 - epsilon * N/K) * N"""

    return float(v * (1.0 - epsilon * customers / market_size) * customers)

def cash_drift(
    customers: float,
    p: float,
    q: float,
    market_size: float,
    v: float,
    epsilon: float,
    chi: float,
    b0: float,
    gamma: float,
    alpha: float,
) -> float:
    """Evaluate deterministic cash drift."""

    net_growth = customer_drift(customers, p, q, market_size, chi)

    # cash flow = revenue - fixed overhead - customer costs - acquisition costs for new customers (if net growth positive)
    return float(
        revenue(customers, market_size, v, epsilon)
        - b0
        - gamma * customers
        - alpha * max(net_growth, 0.0)
    )


def record_row(
    customers: float,
    q: float,
    p: float,
    market_size: float,
    dt: float,
    v: float,
    epsilon: float,
    chi: float,
    b0: float,
    gamma: float,
    alpha: float,
    cash: float,
) -> np.ndarray:
    """
    Build one recorded observation row.
    Layout: [customers, acquired, churned, revenue, burn, cash, latent_q]
    """

    acquired = float(max(acquisition_flow(customers, p, q, market_size), 0.0) * dt)
    churned = float(chi * customers * dt)
    current_revenue = revenue(customers, market_size, v, epsilon)
    burn = float(b0 + gamma * customers + alpha * max(customer_drift(customers, p, q, market_size, chi), 0.0))
    return np.array([customers, acquired, churned, current_revenue, burn, cash, q], dtype=np.float64)


def ode_rhs(
    _time: float,
    state: np.ndarray,
    p: float,
    q: float,
    market_size: float,
    v: float,
    epsilon: float,
    chi: float,
    b0: float,
    gamma: float,
    alpha: float,
) -> np.ndarray:
    """
    Evaluate the deterministic state derivative (dN/dt, dC/dt).

    This is the right hand side of the ODE passed to solve_ivp (describes the noise-free dynamics).
    """

    customers = float(state[0])
    cash = float(state[1])
    if cash <= 0.0:
        return np.array([0.0, 0.0], dtype=np.float64)

    return np.array(
        [
            customer_drift(customers, p, q, market_size, chi),
            cash_drift(customers, p, q, market_size, v, epsilon, chi, b0, gamma, alpha),
        ],
        dtype=np.float64,
    )


def ruin_event(
    _time: float,
    state: np.ndarray,
    p: float,
    q: float,
    market_size: float,
    v: float,
    epsilon: float,
    chi: float,
    b0: float,
    gamma: float,
    alpha: float,
) -> float:
    """
    Detect cash ruin during deterministic integration.

    solve_ivp calls this function at each internal step. When the return value
    crosses zero from above (direction = -1), the solver halts and records the
    crossing time, giving a precise ruin timestamp rather than the nearest
    monthly boundary.

    The unused parameters are required by the solve_ivp events API (they must
    match the signature of _ode_rhs).
    """

    del p, q, market_size, v, epsilon, chi, b0, gamma, alpha
    return float(state[1])


ruin_event.terminal = True # stop integration when ruin_event is triggered
ruin_event.direction = -1 # trigger on crossing zero from above (cash going from positive to non-positive), other direction is infeasible


def simulate(
    p: float,
    q: float,
    kappa: float,
    sigma_q: float,
    K: float,
    v: float,
    epsilon: float,
    chi: float,
    b0: float,
    gamma: float,
    alpha: float,
    sigma_N: float,
    N0: float,
    C0: float,
    T: int,
    dt: float = 1.0 / 12.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Simulate advanced customer growth, cash, and latent virality.

    Each step uses operator splitting:
    - First solve the deterministic ODE exactly with solve_ivp
    - Apply customer noise and q noise on top
    - Adjust cash
    """

    # input validation
    if K <= 0.0:
        raise ValueError("K must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if sigma_q < 0.0 or sigma_N < 0.0:
        raise ValueError("noise intensities must be non-negative.")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must satisfy 0 <= epsilon < 1.")

    rng = rng_from_seed(seed)
    trajectory = np.empty((T + 1, 7), dtype=np.float64) # init trajectory matrix
    customers = float(np.clip(N0, 0.0, K))
    cash = float(C0)
    latent_q = float(max(q, Q_MIN)) # imitation coefficient starts at q and evolves via mean-reverting process
    time = 0.0
    trajectory[0] = record_row(customers, latent_q, p, K, dt, v, epsilon, chi, b0, gamma, alpha, cash)

    ruin_time: float | None = None

    # check for failure to start
    if cash <= 0.0:
        ruin_time = 0.0
        if T > 0:
            trajectory[1:] = trajectory[0]

    for step in range(T):
        if ruin_time is not None:
            break

        # 1. solve ODE system (dN/dt, dC/dt) over one time step
        solve_result = solve_ivp(
            ode_rhs,
            (time, time + dt),
            np.array([customers, cash], dtype=np.float64),
            args=(p, latent_q, K, v, epsilon, chi, b0, gamma, alpha),
            method="RK45",
            events=ruin_event, # stops solver if cash hits 0
        )

        # if startup failed during ODE solve, record ruin time and break loop to end simulation
        if solve_result.t_events[0].size > 0:
            ruin_time = float(solve_result.t_events[0][0])
            customers = float(np.clip(solve_result.y_events[0][0, 0], 0.0, K))
            cash = float(solve_result.y_events[0][0, 1])
            trajectory[step + 1] = record_row(customers, latent_q, p, K, dt, v, epsilon, chi, b0, gamma, alpha, cash)
            if step + 1 < T:
                trajectory[step + 2 :] = trajectory[step + 1]
            break

        # extract customers/cash solution from ODE solve to apply noise on top
        deterministic_customers = float(np.clip(solve_result.y[0, -1], 0.0, K))
        deterministic_cash = float(solve_result.y[1, -1])

        # 2. step the latent q value forward: dq = kappa * (q_mean - q) * dt  +  sigma_q * sqrt(dt) * Z
        latent_q = float(
            max(
                latent_q + kappa * (q - latent_q) * dt + sigma_q * np.sqrt(dt) * rng.standard_normal(),
                Q_MIN,
            )
        )

        # 3. apply noise on top of solved customer count
        stochastic_customers = float(
            np.clip(
                deterministic_customers + sigma_N * np.sqrt(max(deterministic_customers, 0.0) * dt) * rng.standard_normal(),
                0.0,
                K,
            )
        )

        # 4. Adjust cash by the marginal revenue the diff between deterministic and stochastic customer count implies,
        # using the price-compressed marginal revenue at the current N.
        cash = float(
            deterministic_cash
            + (v * (1.0 - epsilon * stochastic_customers / K) - gamma)
            * (stochastic_customers - deterministic_customers)
            * dt
        )
        customers = stochastic_customers
        time += dt
        trajectory[step + 1] = record_row(customers, latent_q, p, K, dt, v, epsilon, chi, b0, gamma, alpha, cash)

        # check for ruin again
        if cash <= 0.0:
            ruin_time = time
            if step + 1 < T:
                trajectory[step + 2 :] = trajectory[step + 1]
            break

    return {
        "model": MODEL_NAME,
        "params": {
            "p": float(p),
            "q": float(q),
            "kappa": float(kappa),
            "sigma_q": float(sigma_q),
            "K": float(K),
            "v": float(v),
            "epsilon": float(epsilon),
            "chi": float(chi),
            "b0": float(b0),
            "gamma": float(gamma),
            "alpha": float(alpha),
            "sigma_N": float(sigma_N),
            "N0": float(N0),
            "C0": float(C0),
            "T": int(T),
            "dt": float(dt),
            "seed": seed,
        },
        "trajectory": trajectory,
        "ruin_time": ruin_time,
        "survived": ruin_time is None,
    }


def batch_simulate(
    n_runs: int,
    base_params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run multiple independent advanced simulations."""

    params: dict[str, Any] = dict(DEFAULT_PARAMS)
    if base_params is not None:
        params.update(base_params)

    rng = rng_from_seed(seed)
    results: list[dict[str, Any]] = []
    for _ in range(n_runs):
        run_params = dict(params)
        run_params["seed"] = int(rng.integers(0, np.iinfo(np.int64).max))
        results.append(simulate(**run_params))
    return results

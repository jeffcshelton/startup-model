from __future__ import annotations

from typing import Any

import numpy as np

MODEL_NAME = "baseline"

# note: each parameter value is measured yearly
DEFAULT_PARAMS: dict[str, float | int] = {
    "p": 0.03, # coefficient of innovation (i.e., how good advertising works)
    "q": 0.38, # coefficient of imitation (i.e., how good word-of-mouth works)
    "K": 50_000.0, # market size (i.e., how many potential customers there are)
    "v": 100.0, # revenue per customer (i.e., how much each customer pays)
    "gamma": 40.0, # cost per customer (i.e., how much it costs to serve each customer)
    "b0": 50_000.0, # fixed burn (i.e., how much it costs to run the business aside from serving customers)
    "sigma_N": 5.0, # noise volatility for customer growth (i.e., how unpredictable customer growth is)
    "N0": 10.0, # initial customers (i.e., how many customers we start with)
    "C0": 2_000_000.0, # initial cash (i.e., how much money we start with)
    "T": 60, # number of time steps to simulate (i.e., how long we run the simulation for)
    "dt": 1.0 / 12.0, # time step size (i.e., how long each time step is in years; 1/12 means monthly steps)
}

# central random number generator
def rng_from_seed(seed: int | None) -> np.random.Generator:
    """Build a random number generator."""

    return np.random.default_rng(seed)

# Bass diffusion model implementation
def growth_drift(customers: float, p: float, q: float, market_size: float) -> float:
    """
    Evaluate the deterministic customer growth drift:
    (p + q*N/K) is the adoption rate,
    (K-N) adjusts growth proportionally to the remaining market
    """

    return float((p + q * customers / market_size) * (market_size - customers))

# Row captures snapshot of the system at a given time step, which we record in the trajectory
def row(
    customers: float,
    acquired: float,
    v: float,
    gamma: float,
    b0: float,
    cash: float,
) -> np.ndarray:
    """
    Build one recorded observation row.
    Layout: [customers, acquired, churned (0 for this model), revenue, burn, cash]
    """

    revenue = float(v * customers) # annualized amount (multiply by dt to get per time step)
    burn = float(b0 + gamma * customers) # annualized amount (multiply by dt to get per time step)
    return np.array([customers, acquired, 0.0, revenue, burn, cash], dtype=np.float64)


def simulate(
    p: float,
    q: float,
    K: float,
    v: float,
    gamma: float,
    b0: float,
    sigma_N: float,
    N0: float,
    C0: float,
    T: int,
    dt: float = 1.0 / 12.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Simulate baseline customer growth and cash dynamics.

    Returns dict containing full trajectory, survival status, and params.
    """

    # input validation
    if K <= 0.0:
        raise ValueError("K must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if sigma_N < 0.0:
        raise ValueError("sigma_N must be non-negative.")
    if v <= gamma: # each customer must be profitable
        raise ValueError("v must be greater than gamma.")

    delta = float(v - gamma) # per-customer margin
    rng = rng_from_seed(seed)
    trajectory = np.empty((T + 1, 6), dtype=np.float64) # initialize trajectory matrix
    customers = float(np.clip(N0, 0.0, K)) # ensure initial customers between 0 and K
    cash = float(C0)
    trajectory[0] = row(customers, 0.0, v, gamma, b0, cash) # initial state

    ruin_time: float | None = None
    for step in range(T):

        # if cash ever runs out, we consider the startup failed/ruined and end the simulation
        if cash <= 0.0:
            ruin_time = float(step * dt)
            trajectory[step + 1 :] = trajectory[step]
            break

        drift = growth_drift(customers, p, q, K) # Bass model customer drift

        # customer count noise/volatility proportional to sqrt of customers
        noise = sigma_N * np.sqrt(max(customers, 0.0) * dt) * rng.standard_normal()

        # advance customers and cash with Euler-Maruyama step
        next_customers = float(np.clip(customers + drift * dt + noise, 0.0, K))
        next_cash = float(cash + (delta * customers - b0) * dt)

        # update trajectory (max(drift, 0) counts acquired customers since 0 churn)
        trajectory[step + 1] = row(next_customers, max(drift, 0.0) * dt, v, gamma, b0, next_cash)

        customers = next_customers
        cash = next_cash

        # check for startup failure again after update
        if cash <= 0.0:
            ruin_time = float((step + 1) * dt)
            if step + 1 < T:
                trajectory[step + 2 :] = trajectory[step + 1]
            break

    return {
        "model": MODEL_NAME,
        "params": {
            "p": float(p),
            "q": float(q),
            "K": float(K),
            "v": float(v),
            "gamma": float(gamma),
            "b0": float(b0),
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
    """
    Run multiple independent baseline simulations.

    Each run uses the same base params with a different random seed for noise.
    """

    params: dict[str, Any] = dict(DEFAULT_PARAMS)
    if base_params is not None:
        params.update(base_params)

    rng = rng_from_seed(seed) # generation of individual seeds is itself seeded for reproducicibility
    results: list[dict[str, Any]] = []
    for _ in range(n_runs):
        run_params = dict(params)
        run_params["seed"] = int(rng.integers(0, np.iinfo(np.int64).max))
        results.append(simulate(**run_params))
    return results

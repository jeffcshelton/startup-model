from __future__ import annotations

from typing import Any

import numpy as np

Q_MIN = 0.001
DEFAULT_PARAMS: dict[str, float | int] = {
    "p": 0.03,
    "q": 0.38,
    "kappa": 1.0,
    "sigma_q": 0.05,
    "K": 50_000.0,
    "v": 100.0,
    "epsilon": 0.1,
    "b0": 50_000.0,
    "gamma": 40.0,
    "alpha": 200.0,
    "sigma_N": 5.0,
    "N0": 10.0,
    "C0": 2_000_000.0,
    "T": 60,
    "dt": 1.0 / 12.0,
}


def _rng_from_seed(seed: int | None) -> np.random.Generator:
    """Build a random number generator.

    Parameters
    ----------
    seed : int or None
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    numpy.random.Generator
        Random number generator for simulation.
    """

    return np.random.default_rng(seed)


def acquisition_flow(customers: float, p: float, q: float, market_size: float) -> float:
    """Evaluate gross customer acquisition before churn.

    Parameters
    ----------
    customers : float
        Current customer count.
    p : float
        Innovation coefficient.
    q : float
        Viral coefficient.
    market_size : float
        Total addressable market.

    Returns
    -------
    float
        Gross acquisition flow.
    """

    return float((p + q * customers / market_size) * (market_size - customers))


def customer_drift(customers: float, p: float, q: float, market_size: float) -> float:
    """Evaluate deterministic customer drift.

    Parameters
    ----------
    customers : float
        Current customer count.
    p : float
        Innovation coefficient.
    q : float
        Viral coefficient.
    market_size : float
        Total addressable market.
    Returns
    -------
    float
        Deterministic customer drift.
    """

    return acquisition_flow(customers, p, q, market_size)


def revenue(customers: float, market_size: float, v: float, epsilon: float) -> float:
    """Evaluate price-compressed revenue.

    Parameters
    ----------
    customers : float
        Current customer count.
    market_size : float
        Total addressable market.
    v : float
        Baseline revenue per customer.
    epsilon : float
        Price-compression coefficient.

    Returns
    -------
    float
        Revenue at the current state.
    """

    return float(v * (1.0 - epsilon * customers / market_size) * customers)


def cash_drift(
    customers: float,
    p: float,
    q: float,
    market_size: float,
    v: float,
    epsilon: float,
    b0: float,
    gamma: float,
    alpha: float,
) -> float:
    """Evaluate deterministic cash drift.

    Parameters
    ----------
    customers : float
        Current customer count.
    p : float
        Innovation coefficient.
    q : float
        Viral coefficient.
    market_size : float
        Total addressable market.
    v : float
        Baseline revenue per customer.
    epsilon : float
        Price-compression coefficient.
    b0 : float
        Fixed burn.
    gamma : float
        Per-customer operating cost.
    alpha : float
        Cost per net new customer.

    Returns
    -------
    float
        Deterministic cash drift.
    """

    net_growth = customer_drift(customers, p, q, market_size)
    return float(
        revenue(customers, market_size, v, epsilon)
        - b0
        - gamma * customers
        - alpha * max(net_growth, 0.0)
    )


def _record_row(
    customers: float,
    q: float,
    p: float,
    market_size: float,
    dt: float,
    v: float,
    epsilon: float,
    b0: float,
    gamma: float,
    alpha: float,
    cash: float,
) -> np.ndarray:
    """Build one recorded observation row.

    Parameters
    ----------
    customers : float
        Customer count.
    q : float
        Current viral coefficient.
    p : float
        Innovation coefficient.
    market_size : float
        Total addressable market.
    dt : float
        Time step.
    v : float
        Baseline revenue per customer.
    epsilon : float
        Price-compression coefficient.
    b0 : float
        Fixed burn.
    gamma : float
        Per-customer operating cost.
    alpha : float
        Cost per net new customer.
    cash : float
        Cash on hand.

    Returns
    -------
    numpy.ndarray
        Float64 observation row.
    """

    acquired = float(max(acquisition_flow(customers, p, q, market_size), 0.0) * dt)
    current_revenue = revenue(customers, market_size, v, epsilon)
    burn = float(b0 + gamma * customers + alpha * max(customer_drift(customers, p, q, market_size), 0.0))
    return np.array([customers, acquired, 0.0, current_revenue, burn, cash, q], dtype=np.float64)


def simulate(
    p: float,
    q: float,
    kappa: float,
    sigma_q: float,
    K: float,
    v: float,
    epsilon: float,
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
    """Simulate customer growth, cash, and latent virality.

    Parameters
    ----------
    p : float
        Innovation coefficient.
    q : float
        Long-run viral coefficient.
    kappa : float
        Mean-reversion speed of latent virality.
    sigma_q : float
        Latent virality volatility.
    K : float
        Total addressable market.
    v : float
        Baseline revenue per customer.
    epsilon : float
        Price-compression coefficient.
    b0 : float
        Fixed burn.
    gamma : float
        Per-customer operating cost.
    alpha : float
        Cost per net new customer.
    sigma_N : float
        Customer-noise intensity.
    N0 : float
        Initial customer count.
    C0 : float
        Initial cash balance.
    T : int
        Number of time steps.
    dt : float, default=1/12
        Time step size.
    seed : int or None, default=None
        Random seed.

    Returns
    -------
    dict of str to Any
        Parameters, trajectory, ruin time, and survival flag.
    """

    if K <= 0.0:
        raise ValueError("K must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if sigma_q < 0.0 or sigma_N < 0.0:
        raise ValueError("noise intensities must be non-negative.")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must satisfy 0 <= epsilon < 1.")

    rng = _rng_from_seed(seed)
    trajectory = np.empty((T + 1, 7), dtype=np.float64)
    customers = float(np.clip(N0, 0.0, K))
    cash = float(C0)
    latent_q = float(max(q, Q_MIN))
    trajectory[0] = _record_row(customers, latent_q, p, K, dt, v, epsilon, b0, gamma, alpha, cash)

    ruin_time: float | None = None
    for step in range(T):
        if cash <= 0.0:
            ruin_time = float(step * dt)
            trajectory[step + 1 :] = trajectory[step]
            break

        drift = customer_drift(customers, p, latent_q, K)
        customer_noise = sigma_N * np.sqrt(max(customers, 0.0) * dt) * rng.standard_normal()
        next_customers = float(np.clip(customers + drift * dt + customer_noise, 0.0, K))
        next_cash = float(cash + cash_drift(customers, p, latent_q, K, v, epsilon, b0, gamma, alpha) * dt)
        next_latent_q = float(
            max(
                latent_q + kappa * (q - latent_q) * dt + sigma_q * np.sqrt(dt) * rng.standard_normal(),
                Q_MIN,
            )
        )
        trajectory[step + 1] = _record_row(next_customers, next_latent_q, p, K, dt, v, epsilon, b0, gamma, alpha, next_cash)

        customers = next_customers
        cash = next_cash
        latent_q = next_latent_q

        if cash <= 0.0:
            ruin_time = float((step + 1) * dt)
            if step + 1 < T:
                trajectory[step + 2 :] = trajectory[step + 1]
            break

    return {
        "params": {
            "p": float(p),
            "q": float(q),
            "kappa": float(kappa),
            "sigma_q": float(sigma_q),
            "K": float(K),
            "v": float(v),
            "epsilon": float(epsilon),
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
    """Run multiple independent simulations.

    Parameters
    ----------
    n_runs : int
        Number of runs.
    base_params : dict of str to Any or None, default=None
        Parameter overrides applied to all runs.
    seed : int or None, default=None
        Seed used to derive per-run seeds.

    Returns
    -------
    list of dict of str to Any
        Simulation results.
    """

    params: dict[str, Any] = dict(DEFAULT_PARAMS)
    if base_params is not None:
        params.update(base_params)

    rng = _rng_from_seed(seed)
    results: list[dict[str, Any]] = []
    for _ in range(n_runs):
        run_params = dict(params)
        run_params["seed"] = int(rng.integers(0, np.iinfo(np.int64).max))
        results.append(simulate(**run_params))
    return results

from __future__ import annotations

from typing import Any

import numpy as np

MODEL_NAME = "baseline"
DEFAULT_PARAMS: dict[str, float | int] = {
    "p": 0.03,
    "q": 0.38,
    "K": 50_000.0,
    "v": 100.0,
    "gamma": 40.0,
    "b0": 50_000.0,
    "sigma_N": 5.0,
    "N0": 10.0,
    "C0": 2_000_000.0,
    "T": 60,
    "dt": 1.0 / 12.0,
}


def _rng_from_seed(seed: int | None) -> np.random.Generator:
    """Build a random number generator."""

    return np.random.default_rng(seed)


def growth_drift(customers: float, p: float, q: float, market_size: float) -> float:
    """Evaluate the deterministic customer growth drift."""

    return float((p + q * customers / market_size) * (market_size - customers))


def _row(
    customers: float,
    acquired: float,
    v: float,
    gamma: float,
    b0: float,
    cash: float,
) -> np.ndarray:
    """Build one recorded observation row."""

    revenue = float(v * customers)
    burn = float(b0 + gamma * customers)
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
    """Simulate baseline customer growth and cash dynamics."""

    if K <= 0.0:
        raise ValueError("K must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if sigma_N < 0.0:
        raise ValueError("sigma_N must be non-negative.")
    if v <= gamma:
        raise ValueError("v must be greater than gamma.")

    delta = float(v - gamma)
    rng = _rng_from_seed(seed)
    trajectory = np.empty((T + 1, 6), dtype=np.float64)
    customers = float(np.clip(N0, 0.0, K))
    cash = float(C0)
    trajectory[0] = _row(customers, 0.0, v, gamma, b0, cash)

    ruin_time: float | None = None
    for step in range(T):
        if cash <= 0.0:
            ruin_time = float(step * dt)
            trajectory[step + 1 :] = trajectory[step]
            break

        drift = growth_drift(customers, p, q, K)
        noise = sigma_N * np.sqrt(max(customers, 0.0) * dt) * rng.standard_normal()
        next_customers = float(np.clip(customers + drift * dt + noise, 0.0, K))
        next_cash = float(cash + (delta * customers - b0) * dt)
        trajectory[step + 1] = _row(next_customers, max(drift, 0.0) * dt, v, gamma, b0, next_cash)

        customers = next_customers
        cash = next_cash

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
    """Run multiple independent baseline simulations."""

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

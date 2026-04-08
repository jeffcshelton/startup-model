from __future__ import annotations

from typing import Any

import numpy as np

from startup_sim.inference.utils import BASELINE_PARAM_ORDER, baseline_theta_to_dict


def _extract_baseline_observables(
    observed: np.ndarray | dict[str, Any],
    *,
    initial_cash: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, float]:
    if isinstance(observed, dict):
        trajectory = np.asarray(observed["trajectory"], dtype=np.float64)
        return trajectory[:, 0], trajectory[:, 3], trajectory[:, 4], trajectory[:, 5], float(trajectory[0, 5])

    observed_array = np.asarray(observed, dtype=np.float64)
    if observed_array.ndim == 2:
        if observed_array.shape[1] < 6:
            raise ValueError("2D observed trajectories must include a cash column or pass initial_cash.")
        return observed_array[:, 0], observed_array[:, 3], observed_array[:, 4], observed_array[:, 5], float(observed_array[0, 5])

    if observed_array.ndim != 1:
        raise ValueError("observed must be a 1D customer path, full trajectory, or result dict.")
    if initial_cash is None:
        raise ValueError("initial_cash is required when observed is a 1D customer path.")
    return observed_array, None, None, None, float(initial_cash)


def _extract_customers_and_cash0(
    observed: np.ndarray | dict[str, Any],
    *,
    initial_cash: float | None = None,
) -> tuple[np.ndarray, float]:
    customers, _, _, _, cash0 = _extract_baseline_observables(observed, initial_cash=initial_cash)
    return customers, cash0


def _gaussian_log_likelihood(observed: np.ndarray, predicted: np.ndarray, sigma: float) -> float:
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("observation sigma must be positive.")
    residual = np.asarray(observed, dtype=np.float64) - np.asarray(predicted, dtype=np.float64)
    sigma_sq = sigma**2
    n = residual.size
    return float(-(n / 2.0) * np.log(2.0 * np.pi * sigma_sq) - 0.5 * np.sum((residual**2) / sigma_sq))


def baseline_cash_path(
    theta: np.ndarray | list[float] | tuple[float, ...] | dict[str, float],
    customers: np.ndarray,
    *,
    initial_cash: float,
    dt: float,
) -> np.ndarray:
    """Reconstruct the baseline cash path from theta and the observed customer path."""

    params = baseline_theta_to_dict(theta) if not isinstance(theta, dict) else theta
    customers = np.asarray(customers, dtype=np.float64)
    cash = np.empty_like(customers, dtype=np.float64)
    cash[0] = float(initial_cash)
    delta = float(params["v"] - params["gamma"])
    for idx in range(len(customers) - 1):
        cash[idx + 1] = cash[idx] + (delta * customers[idx] - float(params["b0"])) * float(dt)
    return cash


def baseline_survival_indicator(
    theta: np.ndarray | list[float] | tuple[float, ...] | dict[str, float],
    observed: np.ndarray | dict[str, Any],
    *,
    dt: float = 1.0 / 12.0,
    initial_cash: float | None = None,
) -> float:
    """Return 0.0 for survival through T and -inf for ruin before T."""

    customers, cash0 = _extract_customers_and_cash0(observed, initial_cash=initial_cash)
    cash = baseline_cash_path(theta, customers, initial_cash=cash0, dt=dt)
    return 0.0 if bool(np.all(cash > 0.0)) else float("-inf")


def baseline_log_likelihood(
    theta: np.ndarray | list[float] | tuple[float, ...] | dict[str, float],
    observed: np.ndarray | dict[str, Any],
    *,
    dt: float = 1.0 / 12.0,
    initial_cash: float | None = None,
    revenue_obs_sigma: float | None = None,
    burn_obs_sigma: float | None = None,
    cash_obs_sigma: float | None = None,
) -> float:
    """Evaluate the approximate closed-form baseline trajectory log-likelihood."""

    params = baseline_theta_to_dict(theta) if not isinstance(theta, dict) else theta
    customers, revenue_obs, burn_obs, cash_obs, cash0 = _extract_baseline_observables(observed, initial_cash=initial_cash)
    if customers.ndim != 1 or customers.size < 2:
        raise ValueError("customers must be a 1D array with at least two observations.")
    if params["K"] <= 0.0 or params["sigma_N"] <= 0.0 or dt <= 0.0:
        return float("-inf")
    if params["v"] <= params["gamma"]:
        return float("-inf")
    if np.any(customers[:-1] <= 0.0):
        return float("-inf")

    survival = baseline_survival_indicator(params, customers, dt=dt, initial_cash=cash0)
    if not np.isfinite(survival):
        return float("-inf")

    current = customers[:-1]
    nxt = customers[1:]
    drift = (params["p"] + params["q"] * current / params["K"]) * (params["K"] - current)
    residual = nxt - current - drift * dt
    sigma_sq = float(params["sigma_N"]) ** 2
    T = customers.size - 1

    loglike = -(T / 2.0) * np.log(2.0 * np.pi * dt)
    loglike -= 0.5 * np.sum(np.log(sigma_sq * current))
    loglike -= 0.5 / (sigma_sq * dt) * np.sum((residual**2) / current)

    # Revenue and cash are deterministic given theta and the customer path in the simulator.
    # We condition on their observed histories with small Gaussian observation noise so HMC
    # can use those time series without collapsing onto an exact lower-dimensional manifold.
    if revenue_obs is not None and revenue_obs_sigma is not None:
        predicted_revenue = float(params["v"]) * customers
        loglike += _gaussian_log_likelihood(revenue_obs, predicted_revenue, revenue_obs_sigma)
    if burn_obs is not None and burn_obs_sigma is not None:
        predicted_burn = float(params["b0"]) + float(params["gamma"]) * customers
        loglike += _gaussian_log_likelihood(burn_obs, predicted_burn, burn_obs_sigma)
    if cash_obs is not None and cash_obs_sigma is not None:
        predicted_cash = baseline_cash_path(params, customers, initial_cash=cash0, dt=dt)
        loglike += _gaussian_log_likelihood(cash_obs, predicted_cash, cash_obs_sigma)

    return float(loglike + survival)


def baseline_parameter_names() -> tuple[str, ...]:
    return BASELINE_PARAM_ORDER

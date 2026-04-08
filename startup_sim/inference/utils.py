from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import stats

from startup_sim import advanced as advanced_model
from startup_sim import baseline as baseline_model
from startup_sim.inference.config import InferenceConfig

BASELINE_PARAM_ORDER = ("p", "q", "K", "v", "gamma", "b0", "sigma_N")
ADVANCED_PARAM_ORDER = (
    "p",
    "q",
    "kappa",
    "sigma_q",
    "K",
    "v",
    "epsilon",
    "chi",
    "b0",
    "gamma",
    "alpha",
    "sigma_N",
)
BASELINE_OBSERVABLES = ("cash", "revenue", "customers", "delta_n_plus", "delta_n_minus")


@dataclass(slots=True)
class TrialData:
    theta_true: dict[str, float]
    observed_result: dict[str, Any]
    forecast_result: dict[str, Any]
    observed_observables: np.ndarray
    future_observables: np.ndarray


def set_global_seed(seed: int) -> np.random.Generator:
    """Create the canonical NumPy RNG for one experiment."""

    return np.random.default_rng(seed)


def observable_matrix(trajectory: np.ndarray) -> np.ndarray:
    """Return the five observed dimensions used by inference and evaluation."""

    trajectory = np.asarray(trajectory, dtype=np.float64)
    customers = trajectory[:, 0]
    cash = trajectory[:, 5]
    revenue = trajectory[:, 3]
    delta_customers = np.diff(customers, prepend=customers[0])
    delta_n_plus = np.maximum(delta_customers, 0.0)
    delta_n_minus = np.maximum(-delta_customers, 0.0)
    return np.stack([cash, revenue, customers, delta_n_plus, delta_n_minus], axis=-1)


def baseline_theta_to_dict(theta: np.ndarray | list[float] | tuple[float, ...]) -> dict[str, float]:
    values = np.asarray(theta, dtype=np.float64)
    if values.shape != (len(BASELINE_PARAM_ORDER),):
        raise ValueError("baseline theta has unexpected shape.")
    return {name: float(value) for name, value in zip(BASELINE_PARAM_ORDER, values, strict=True)}


def dict_to_theta(params: dict[str, float], *, model: str) -> np.ndarray:
    order = BASELINE_PARAM_ORDER if model == "baseline" else ADVANCED_PARAM_ORDER
    return np.asarray([float(params[name]) for name in order], dtype=np.float64)


def prior_sampler(model: str) -> Callable[[np.random.Generator, int], np.ndarray]:
    """Return a vectorized prior sampler."""

    if model == "baseline":
        def sample(rng: np.random.Generator, n: int) -> np.ndarray:
            return np.column_stack(
                [
                    rng.beta(1.5, 30.0, size=n),
                    rng.beta(3.0, 10.0, size=n),
                    rng.lognormal(8.5, 0.8, size=n),
                    rng.lognormal(4.6, 0.4, size=n),
                    rng.lognormal(4.2, 0.5, size=n),
                    rng.lognormal(10.8, 0.6, size=n),
                    np.abs(rng.normal(0.0, 50.0, size=n)),
                ]
            ).astype(np.float64)
        return sample

    if model == "advanced":
        def sample(rng: np.random.Generator, n: int) -> np.ndarray:
            # Extra advanced-model priors are weakly informative and centered on simulator defaults.
            return np.column_stack(
                [
                    rng.beta(1.5, 30.0, size=n),
                    rng.beta(3.0, 10.0, size=n),
                    rng.lognormal(np.log(1.0), 0.5, size=n),
                    np.abs(rng.normal(0.0, 0.1, size=n)),
                    rng.lognormal(8.5, 0.8, size=n),
                    rng.lognormal(4.6, 0.4, size=n),
                    rng.beta(2.0, 18.0, size=n),
                    rng.lognormal(np.log(0.02), 0.5, size=n),
                    rng.lognormal(10.8, 0.6, size=n),
                    rng.lognormal(4.2, 0.5, size=n),
                    rng.lognormal(np.log(200.0), 0.5, size=n),
                    np.abs(rng.normal(0.0, 50.0, size=n)),
                ]
            ).astype(np.float64)
        return sample

    raise ValueError(f"unsupported model {model!r}")


def prior_variances(model: str, n_mc: int = 200_000, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = prior_sampler(model)(rng, n_mc)
    order = BASELINE_PARAM_ORDER if model == "baseline" else ADVANCED_PARAM_ORDER
    return {name: float(np.var(samples[:, idx], ddof=1)) for idx, name in enumerate(order)}


def simulate_from_theta(
    theta: dict[str, float],
    *,
    model: str,
    cfg: InferenceConfig,
    seed: int,
    N0: float | None = None,
    C0: float | None = None,
    T: int | None = None,
) -> dict[str, Any]:
    """Run one simulator with explicit parameter dict."""

    if model == "baseline":
        defaults = dict(baseline_model.DEFAULT_PARAMS)
        defaults.update(theta)
        if float(defaults["v"]) <= float(defaults["gamma"]):
            raise ValueError("baseline parameters require v > gamma.")
        defaults["T"] = int(cfg.observation_steps if T is None else T)
        defaults["dt"] = float(cfg.dt)
        defaults["seed"] = int(seed)
        if N0 is not None:
            defaults["N0"] = float(N0)
        if C0 is not None:
            defaults["C0"] = float(C0)
        return baseline_model.simulate(**defaults)

    if model == "advanced":
        defaults = dict(advanced_model.DEFAULT_PARAMS)
        defaults.update(theta)
        defaults["T"] = int(cfg.observation_steps if T is None else T)
        defaults["dt"] = float(cfg.dt)
        defaults["seed"] = int(seed)
        if N0 is not None:
            defaults["N0"] = float(N0)
        if C0 is not None:
            defaults["C0"] = float(C0)
        return advanced_model.simulate(**defaults)

    raise ValueError(f"unsupported model {model!r}")


def split_observed_and_future(
    theta: dict[str, float],
    *,
    model: str,
    cfg: InferenceConfig,
    seed: int,
) -> TrialData:
    """Simulate one observed window and one future window from the same parameters."""

    observed = simulate_from_theta(theta, model=model, cfg=cfg, seed=seed, T=cfg.observation_steps)
    last = observed["trajectory"][-1]
    future = simulate_from_theta(
        theta,
        model=model,
        cfg=cfg,
        seed=seed + 1,
        N0=float(last[0]),
        C0=float(last[5]),
        T=cfg.forecast_steps,
    )
    return TrialData(
        theta_true={key: float(value) for key, value in theta.items()},
        observed_result=observed,
        forecast_result=future,
        observed_observables=observable_matrix(observed["trajectory"]),
        future_observables=observable_matrix(future["trajectory"]),
    )


def sample_surviving_trial(model: str, cfg: InferenceConfig, seed: int, max_attempts: int = 10_000) -> TrialData:
    """Sample from the prior until the observed trajectory survives through T."""

    rng = np.random.default_rng(seed)
    sampler = prior_sampler(model)
    order = BASELINE_PARAM_ORDER if model == "baseline" else ADVANCED_PARAM_ORDER
    for attempt in range(max_attempts):
        theta_values = sampler(rng, 1)[0]
        theta = {name: float(value) for name, value in zip(order, theta_values, strict=True)}
        try:
            trial = split_observed_and_future(theta, model=model, cfg=cfg, seed=int(rng.integers(0, 2**31 - 1)))
        except ValueError:
            continue
        if trial.observed_result["survived"]:
            return trial
    raise RuntimeError("failed to draw a surviving trajectory from the prior.")


def kde_logpdf(samples: np.ndarray, point: np.ndarray) -> float:
    """Evaluate a Gaussian KDE log density at one point."""

    samples = np.asarray(samples, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    kde = stats.gaussian_kde(samples.T)
    density = float(np.asarray(kde(point)).reshape(()))
    tiny = np.finfo(np.float64).tiny
    return float(np.log(max(density, tiny)))

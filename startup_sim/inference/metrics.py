from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from startup_sim.inference.utils import kde_logpdf, prior_variances


def energy_score_ensemble(predictive: np.ndarray, truth: np.ndarray) -> float:
    predictive = np.asarray(predictive, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    first = np.mean(np.linalg.norm(predictive - truth, axis=1))
    diffs = predictive[:, None, :] - predictive[None, :, :]
    second = 0.5 * np.mean(np.linalg.norm(diffs, axis=-1))
    return float(first - second)


def energy_score_by_dimension(predictive: np.ndarray, truth: np.ndarray) -> dict[int, float]:
    predictive = np.asarray(predictive, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    scores: dict[int, float] = {}
    for dim in range(predictive.shape[-1]):
        flattened_predictive = predictive[:, :, dim]
        flattened_truth = truth[:, dim]
        scores[dim] = energy_score_ensemble(flattened_predictive, flattened_truth)
    return scores


def posterior_nll_snpe(log_prob: float) -> float:
    return float(-log_prob)


def posterior_nll_mcmc_kde(samples: np.ndarray, theta_true: np.ndarray) -> float:
    return float(-kde_logpdf(samples, theta_true))


def sbc_rank(samples: np.ndarray, theta_true: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    theta_true = np.asarray(theta_true, dtype=np.float64)
    return np.sum(samples < theta_true[None, :], axis=0)


def pit_values(predictive: np.ndarray, truth: np.ndarray) -> np.ndarray:
    predictive = np.asarray(predictive, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return np.mean(predictive <= truth[None, ...], axis=0)


def prediction_interval_width(predictive: np.ndarray, lower: float = 0.1, upper: float = 0.9) -> np.ndarray:
    predictive = np.asarray(predictive, dtype=np.float64)
    return np.quantile(predictive, upper, axis=0) - np.quantile(predictive, lower, axis=0)


def prediction_interval_coverage(predictive: np.ndarray, truth: np.ndarray, lower: float = 0.1, upper: float = 0.9) -> np.ndarray:
    predictive = np.asarray(predictive, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    lo = np.quantile(predictive, lower, axis=0)
    hi = np.quantile(predictive, upper, axis=0)
    return ((truth >= lo) & (truth <= hi)).astype(np.float64)


def posterior_std(samples: np.ndarray, names: Iterable[str]) -> dict[str, float]:
    samples = np.asarray(samples, dtype=np.float64)
    return {name: float(np.std(samples[:, idx], ddof=1)) for idx, name in enumerate(names)}


def posterior_variance_ratio(samples: np.ndarray, names: Iterable[str], *, model: str) -> dict[str, float]:
    samples = np.asarray(samples, dtype=np.float64)
    prior_vars = prior_variances(model)
    ratios: dict[str, float] = {}
    for idx, name in enumerate(names):
        post_var = float(np.var(samples[:, idx], ddof=1))
        ratios[name] = post_var / max(prior_vars[name], np.finfo(np.float64).tiny)
    return ratios


def classify_regime(result: dict) -> str:
    """Classify a ground-truth run into surviving, slow-bleed ruin, or growth-induced ruin."""

    if result["survived"]:
        return "surviving"
    trajectory = np.asarray(result["trajectory"], dtype=np.float64)
    customers = trajectory[:, 0]
    deltas = np.diff(customers)
    active = deltas[: max(1, min(len(deltas), 6))]
    # Simple, explicit heuristic: positive customer momentum into ruin indicates growth-induced ruin.
    if active.size and float(np.mean(deltas[-min(len(deltas), 6) :])) > 0.0:
        return "growth_induced_ruin"
    return "slow_bleed_ruin"


def ruin_probability(predictive_observables: np.ndarray) -> float:
    predictive_observables = np.asarray(predictive_observables, dtype=np.float64)
    cash = predictive_observables[:, :, 0]
    ruined = np.any(cash <= 0.0, axis=1)
    return float(np.mean(ruined))


@dataclass(slots=True)
class CalibrationBundle:
    sbc_ranks: list[np.ndarray]
    pit: list[np.ndarray]

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from startup_sim.inference.utils import BASELINE_OBSERVABLES


def save_sbc_rank_histograms(ranks: np.ndarray, names: list[str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(names), 1, figsize=(8, 3 * len(names)), squeeze=False)
    for idx, name in enumerate(names):
        axis = axes[idx, 0]
        axis.hist(ranks[:, idx], bins=20, color="#2563eb", alpha=0.85)
        axis.set_title(f"SBC Rank Histogram: {name}")
        axis.set_xlabel("Rank")
        axis.set_ylabel("Count")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_pit_histograms(pit_values: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(BASELINE_OBSERVABLES), 1, figsize=(8, 3 * len(BASELINE_OBSERVABLES)), squeeze=False)
    for idx, name in enumerate(BASELINE_OBSERVABLES):
        axis = axes[idx, 0]
        axis.hist(pit_values[:, idx], bins=20, range=(0.0, 1.0), color="#d97706", alpha=0.85)
        axis.set_title(f"PIT Histogram: {name}")
        axis.set_xlabel("PIT")
        axis.set_ylabel("Count")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_posterior_predictive_overlay(
    observed: np.ndarray,
    predictive: np.ndarray,
    path: str | Path,
    *,
    central_mass: float = 0.8,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha = (1.0 - central_mass) / 2.0
    lo = np.quantile(predictive, alpha, axis=0)
    hi = np.quantile(predictive, 1.0 - alpha, axis=0)

    figure, axes = plt.subplots(len(BASELINE_OBSERVABLES), 1, figsize=(10, 3 * len(BASELINE_OBSERVABLES)), squeeze=False)
    time_axis = np.arange(observed.shape[0])
    for idx, name in enumerate(BASELINE_OBSERVABLES):
        axis = axes[idx, 0]
        axis.fill_between(time_axis, lo[:, idx], hi[:, idx], color="#93c5fd", alpha=0.6, label="central predictive region")
        axis.plot(time_axis, observed[:, idx], color="#111827", linewidth=2.0, label="observed")
        axis.set_title(f"Posterior Predictive Check: {name}")
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path

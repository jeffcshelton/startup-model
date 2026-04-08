from __future__ import annotations

import concurrent.futures
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from startup_sim.inference.config import InferenceConfig
from startup_sim.inference.mcmc import posterior_array, posterior_predictive_baseline, run_baseline_nuts
from startup_sim.inference.metrics import (
    classify_regime,
    energy_score_by_dimension,
    pit_values,
    posterior_nll_mcmc_kde,
    posterior_nll_snpe,
    posterior_std,
    posterior_variance_ratio,
    prediction_interval_coverage,
    prediction_interval_width,
    ruin_probability,
    sbc_rank,
)
from startup_sim.inference.plotting import save_pit_histograms, save_posterior_predictive_overlay, save_sbc_rank_histograms
from startup_sim.inference.snpe import posterior_log_prob, posterior_predictive, sample_posterior, train_snpe
from startup_sim.inference.utils import ADVANCED_PARAM_ORDER, BASELINE_OBSERVABLES, BASELINE_PARAM_ORDER, dict_to_theta, sample_surviving_trial


@dataclass(slots=True)
class TrialSummary:
    energy_scores: dict[str, float]
    parameter_nll: float
    posterior_std: dict[str, float]
    posterior_variance_ratio: dict[str, float]
    interval_width: dict[str, float]
    interval_coverage: dict[str, float]
    regime: str
    predictive_ruin_probability: float
    empirical_ruin: float
    inference_seconds: float
    training_seconds: float
    simulator_calls: int
    amortized_query_seconds: float
    sbc_rank: list[int]
    pit_values: list[list[float]]


@dataclass(slots=True)
class EvaluationSummary:
    method: str
    model: str
    n_trials: int
    average_energy_scores: dict[str, float]
    average_parameter_nll: float
    average_posterior_std: dict[str, float]
    average_variance_ratio: dict[str, float]
    average_interval_width: dict[str, float]
    average_interval_coverage: dict[str, float]
    average_inference_seconds: float
    average_training_seconds: float
    average_simulator_calls: float
    average_amortized_query_seconds: float
    failure_mode_summary: dict[str, dict[str, float]]
    summary_path: Path
    sbc_plot_path: Path
    pit_plot_path: Path


def _param_order(model: str) -> tuple[str, ...]:
    return BASELINE_PARAM_ORDER if model == "baseline" else ADVANCED_PARAM_ORDER


def _energy_named(scores: dict[int, float]) -> dict[str, float]:
    return {name: float(scores[idx]) for idx, name in enumerate(BASELINE_OBSERVABLES)}


def _mean_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    keys = items[0].keys()
    return {key: float(np.mean([item[key] for item in items])) for key in keys}


def _run_single_trial_with_snpe(
    *,
    snpe_result,
    model: str,
    cfg: InferenceConfig,
    seed: int,
) -> TrialSummary:
    trial = sample_surviving_trial(model, cfg, seed)
    posterior_start = time.perf_counter()
    samples = sample_posterior(
        snpe_result,
        trial.observed_observables,
        num_samples=cfg.snpe_posterior_samples,
        seed=seed + 17,
    )
    amortized_seconds = time.perf_counter() - posterior_start
    predictive = posterior_predictive(snpe_result, samples, trial.observed_result, cfg=cfg, seed=seed + 33)
    theta_true = dict_to_theta(trial.theta_true, model=model)
    parameter_nll = posterior_nll_snpe(posterior_log_prob(snpe_result, theta_true, trial.observed_observables))
    pit = pit_values(predictive, trial.future_observables)
    piw = prediction_interval_width(predictive)
    coverage = prediction_interval_coverage(predictive, trial.future_observables)
    order = list(_param_order(model))
    regime = classify_regime(trial.forecast_result)
    return TrialSummary(
        energy_scores=_energy_named(energy_score_by_dimension(predictive, trial.future_observables)),
        parameter_nll=parameter_nll,
        posterior_std=posterior_std(samples, order),
        posterior_variance_ratio=posterior_variance_ratio(samples, order, model=model),
        interval_width={name: float(np.mean(piw[:, idx])) for idx, name in enumerate(BASELINE_OBSERVABLES)},
        interval_coverage={name: float(np.mean(coverage[:, idx])) for idx, name in enumerate(BASELINE_OBSERVABLES)},
        regime=regime,
        predictive_ruin_probability=ruin_probability(predictive),
        empirical_ruin=float(not trial.forecast_result["survived"]),
        inference_seconds=amortized_seconds,
        training_seconds=0.0,
        simulator_calls=cfg.snpe_posterior_samples,
        amortized_query_seconds=amortized_seconds,
        sbc_rank=sbc_rank(samples[: cfg.sbc_posterior_samples], theta_true).astype(int).tolist(),
        pit_values=np.asarray(pit, dtype=np.float64).tolist(),
    )


def _run_single_trial_with_mcmc(
    *,
    cfg: InferenceConfig,
    seed: int,
    trace_output_dir: str | Path | None = None,
) -> TrialSummary:
    trial = None
    for attempt in range(1_000):
        candidate = sample_surviving_trial("baseline", cfg, seed + attempt)
        observed_customers = np.asarray(candidate.observed_result["trajectory"], dtype=np.float64)[:-1, 0]
        if np.all(observed_customers > 0.0):
            trial = candidate
            break
    if trial is None:
        raise RuntimeError("failed to draw a baseline trial supported by the closed-form MCMC likelihood.")
    result = run_baseline_nuts(trial.observed_result, cfg=cfg, seed=seed + 11, output_dir=trace_output_dir)
    samples = posterior_array(result)
    predictive = posterior_predictive_baseline(
        result,
        trial.observed_result,
        cfg=cfg,
        seed=seed + 29,
        num_samples=cfg.snpe_posterior_samples,
        show_progress=cfg.mcmc_trial_workers <= 1,
    )
    theta_true = dict_to_theta(trial.theta_true, model="baseline")
    pit = pit_values(predictive, trial.future_observables)
    piw = prediction_interval_width(predictive)
    coverage = prediction_interval_coverage(predictive, trial.future_observables)
    regime = classify_regime(trial.forecast_result)
    return TrialSummary(
        energy_scores=_energy_named(energy_score_by_dimension(predictive, trial.future_observables)),
        parameter_nll=posterior_nll_mcmc_kde(samples, theta_true),
        posterior_std=posterior_std(samples, BASELINE_PARAM_ORDER),
        posterior_variance_ratio=posterior_variance_ratio(samples, BASELINE_PARAM_ORDER, model="baseline"),
        interval_width={name: float(np.mean(piw[:, idx])) for idx, name in enumerate(BASELINE_OBSERVABLES)},
        interval_coverage={name: float(np.mean(coverage[:, idx])) for idx, name in enumerate(BASELINE_OBSERVABLES)},
        regime=regime,
        predictive_ruin_probability=ruin_probability(predictive),
        empirical_ruin=float(not trial.forecast_result["survived"]),
        inference_seconds=result.elapsed_seconds,
        training_seconds=0.0,
        simulator_calls=0,
        amortized_query_seconds=result.elapsed_seconds,
        sbc_rank=sbc_rank(samples[: cfg.sbc_posterior_samples], theta_true).astype(int).tolist(),
        pit_values=np.asarray(pit, dtype=np.float64).tolist(),
    )


def _mcmc_trial_worker_count(cfg: InferenceConfig) -> int:
    if cfg.mcmc_trial_workers > 0:
        return cfg.mcmc_trial_workers
    available = os.cpu_count() or 1
    per_trial = max(cfg.mcmc_num_chains, 1)
    # Leave headroom for Python overhead and posterior predictive simulation.
    return max(1, min(4, available // per_trial))


def run_evaluation_study(
    *,
    method: str,
    model: str,
    cfg: InferenceConfig,
    seed: int,
    output_dir: str | Path | None = None,
) -> EvaluationSummary:
    method = method.lower()
    model = model.lower()
    if method == "mcmc" and model != "baseline":
        raise ValueError("MCMC is only supported for the baseline model.")

    base_output = Path(output_dir or cfg.outputs_dir / f"{method}_{model}")
    base_output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    trial_summaries: list[TrialSummary] = []

    snpe_result = None
    training_seconds = 0.0
    if method == "snpe":
        snpe_result = train_snpe(model=model, cfg=cfg, seed=seed + 5, output_dir=base_output)
        training_seconds = snpe_result.elapsed_seconds

    trial_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(cfg.n_evaluation_trials)]
    if method == "mcmc":
        worker_count = _mcmc_trial_worker_count(cfg)
        if worker_count == 1:
            for trial_index, trial_seed in enumerate(tqdm(trial_seeds, desc=f"Evaluation ({method}, {model})")):
                summary = _run_single_trial_with_mcmc(
                    cfg=cfg,
                    seed=trial_seed,
                    trace_output_dir=base_output / "mcmc_traces" / f"trial_{trial_index:04d}",
                )
                trial_summaries.append(summary)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _run_single_trial_with_mcmc,
                        cfg=cfg,
                        seed=trial_seed,
                        trace_output_dir=base_output / "mcmc_traces" / f"trial_{trial_index:04d}",
                    )
                    for trial_index, trial_seed in enumerate(trial_seeds)
                ]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"Evaluation ({method}, {model})",
                ):
                    trial_summaries.append(future.result())
    else:
        for trial_seed in tqdm(trial_seeds, desc=f"Evaluation ({method}, {model})"):
            summary = _run_single_trial_with_snpe(snpe_result=snpe_result, model=model, cfg=cfg, seed=trial_seed)
            summary.training_seconds = training_seconds
            trial_summaries.append(summary)

    energy = _mean_dicts([trial.energy_scores for trial in trial_summaries])
    nll = float(np.mean([trial.parameter_nll for trial in trial_summaries]))
    post_std = _mean_dicts([trial.posterior_std for trial in trial_summaries])
    var_ratio = _mean_dicts([trial.posterior_variance_ratio for trial in trial_summaries])
    piw = _mean_dicts([trial.interval_width for trial in trial_summaries])
    coverage = _mean_dicts([trial.interval_coverage for trial in trial_summaries])
    avg_inference = float(np.mean([trial.inference_seconds for trial in trial_summaries]))
    avg_training = float(np.mean([trial.training_seconds for trial in trial_summaries]))
    avg_calls = float(np.mean([trial.simulator_calls for trial in trial_summaries]))
    avg_amortized = float(np.mean([trial.amortized_query_seconds for trial in trial_summaries]))

    failure_modes: dict[str, dict[str, float]] = {}
    for regime in ("surviving", "slow_bleed_ruin", "growth_induced_ruin"):
        group = [trial for trial in trial_summaries if trial.regime == regime]
        if not group:
            continue
        failure_modes[regime] = {
            "predictive_ruin_probability": float(np.mean([trial.predictive_ruin_probability for trial in group])),
            "empirical_ruin_rate": float(np.mean([trial.empirical_ruin for trial in group])),
        }

    ranks = np.asarray([trial.sbc_rank for trial in trial_summaries], dtype=np.int64)
    pits = np.concatenate(
        [np.asarray(trial.pit_values, dtype=np.float64) for trial in trial_summaries],
        axis=0,
    )
    sbc_plot = save_sbc_rank_histograms(ranks, list(_param_order(model)), base_output / "sbc_hist.png")
    pit_plot = save_pit_histograms(pits, base_output / "pit_hist.png")

    summary = EvaluationSummary(
        method=method,
        model=model,
        n_trials=cfg.n_evaluation_trials,
        average_energy_scores=energy,
        average_parameter_nll=nll,
        average_posterior_std=post_std,
        average_variance_ratio=var_ratio,
        average_interval_width=piw,
        average_interval_coverage=coverage,
        average_inference_seconds=avg_inference,
        average_training_seconds=avg_training,
        average_simulator_calls=avg_calls,
        average_amortized_query_seconds=avg_amortized,
        failure_mode_summary=failure_modes,
        summary_path=base_output / "summary.json",
        sbc_plot_path=sbc_plot,
        pit_plot_path=pit_plot,
    )

    serializable = asdict(summary)
    serializable["summary_path"] = str(summary.summary_path)
    serializable["sbc_plot_path"] = str(summary.sbc_plot_path)
    serializable["pit_plot_path"] = str(summary.pit_plot_path)
    summary.summary_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return summary


def save_observed_posterior_predictive_check(
    observed_observables: np.ndarray,
    predictive_observables: np.ndarray,
    *,
    path: str | Path,
    central_mass: float,
) -> Path:
    return save_posterior_predictive_overlay(
        observed_observables,
        predictive_observables,
        path,
        central_mass=central_mass,
    )

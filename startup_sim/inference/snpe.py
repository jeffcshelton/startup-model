from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from startup_sim.inference.config import InferenceConfig
from startup_sim.inference.utils import ADVANCED_PARAM_ORDER, BASELINE_PARAM_ORDER, observable_matrix, prior_sampler, simulate_from_theta


def _require_torch_and_sbi():
    try:
        import torch
        import torch.nn as nn
        from sbi.inference import SNPE
        from sbi.utils.user_input_checks import process_prior
        try:
            from sbi.neural_nets import posterior_nn
        except ImportError:
            from sbi.utils import posterior_nn
    except Exception as exc:  # pragma: no cover - exercised only when deps exist
        raise ImportError("SNPE requires torch and sbi to be installed.") from exc
    return torch, nn, SNPE, posterior_nn, process_prior


def _parameter_order(model: str) -> tuple[str, ...]:
    return BASELINE_PARAM_ORDER if model == "baseline" else ADVANCED_PARAM_ORDER


def _prior_components(model: str):  # pragma: no cover - depends on torch/sbi
    torch, _, _, _, _ = _require_torch_and_sbi()
    td = torch.distributions
    tensor = lambda value: torch.tensor([float(value)], dtype=torch.float32)
    if model == "baseline":
        return [
            td.Beta(tensor(1.5), tensor(30.0)),
            td.Beta(tensor(3.0), tensor(10.0)),
            td.LogNormal(tensor(8.5), tensor(0.8)),
            td.LogNormal(tensor(4.6), tensor(0.4)),
            td.LogNormal(tensor(4.2), tensor(0.5)),
            td.LogNormal(tensor(10.8), tensor(0.6)),
            td.HalfNormal(tensor(50.0)),
        ]
    return [
        td.Beta(tensor(1.5), tensor(30.0)),
        td.Beta(tensor(3.0), tensor(10.0)),
        td.LogNormal(tensor(np.log(1.0)), tensor(0.5)),
        td.HalfNormal(tensor(0.1)),
        td.LogNormal(tensor(8.5), tensor(0.8)),
        td.LogNormal(tensor(4.6), tensor(0.4)),
        td.Beta(tensor(2.0), tensor(18.0)),
        td.LogNormal(tensor(np.log(0.02)), tensor(0.5)),
        td.LogNormal(tensor(10.8), tensor(0.6)),
        td.LogNormal(tensor(4.2), tensor(0.5)),
        td.LogNormal(tensor(np.log(200.0)), tensor(0.5)),
        td.HalfNormal(tensor(50.0)),
    ]


def _make_sbi_prior(model: str):  # pragma: no cover - depends on torch/sbi
    _, _, _, _, process_prior = _require_torch_and_sbi()
    prior, _, _ = process_prior(_prior_components(model))
    return prior


def _embedding_class():  # pragma: no cover - depends on torch
    _, nn, _, _, _ = _require_torch_and_sbi()

    class LSTMTrajectoryEmbedding(nn.Module):
        def __init__(self, input_size: int, embedding_dim: int):
            super().__init__()
            self.module = nn.LSTM(input_size=input_size, hidden_size=embedding_dim, batch_first=True)
            self.projection = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )

        def forward(self, x):
            outputs, _ = self.module(x)
            return self.projection(outputs[:, -1, :])

    return LSTMTrajectoryEmbedding


@dataclass(slots=True)
class SNPEResult:
    model: str
    posterior: Any
    training_theta: np.ndarray
    training_x: np.ndarray
    training_losses: list[float]
    validation_losses: list[float]
    elapsed_seconds: float
    training_curve_path: Path


def _simulate_surviving_dataset(model: str, cfg: InferenceConfig, seed: int, n_simulations: int) -> tuple[np.ndarray, np.ndarray]:
    sampler = prior_sampler(model)
    rng = np.random.default_rng(seed)
    order = _parameter_order(model)
    theta_rows: list[np.ndarray] = []
    x_rows: list[np.ndarray] = []
    progress = tqdm(total=n_simulations, desc=f"SNPE simulations ({model})", leave=False)
    while len(theta_rows) < n_simulations:
        theta = sampler(rng, 1)[0]
        theta_dict = {name: float(value) for name, value in zip(order, theta, strict=True)}
        try:
            run = simulate_from_theta(theta_dict, model=model, cfg=cfg, seed=int(rng.integers(0, 2**31 - 1)))
        except ValueError:
            continue
        if not run["survived"]:
            continue
        theta_rows.append(theta.astype(np.float64))
        x_rows.append(observable_matrix(run["trajectory"]).astype(np.float32))
        progress.update(1)
    progress.close()
    return np.stack(theta_rows, axis=0), np.stack(x_rows, axis=0)


def _extract_losses(density_estimator: Any) -> tuple[list[float], list[float]]:  # pragma: no cover - depends on sbi internals
    summary = getattr(density_estimator, "_summary", {})
    training = [float(x) for x in summary.get("training_log_probs", [])]
    validation = [float(x) for x in summary.get("validation_log_probs", [])]
    if training:
        training = [-value for value in training]
    if validation:
        validation = [-value for value in validation]
    return training, validation


def _plot_losses(training_losses: list[float], validation_losses: list[float], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    if training_losses:
        axis.plot(training_losses, label="train", linewidth=2)
    if validation_losses:
        axis.plot(validation_losses, label="validation", linewidth=2)
    axis.set_title("SNPE Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Negative Log Likelihood")
    handles, labels = axis.get_legend_handles_labels()
    if labels:
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _valid_theta_mask(model: str, samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if model != "baseline":
        return np.ones(samples.shape[0], dtype=bool)
    order = _parameter_order(model)
    v_idx = order.index("v")
    gamma_idx = order.index("gamma")
    return np.isfinite(samples).all(axis=1) & (samples[:, v_idx] > samples[:, gamma_idx])


def train_snpe(
    *,
    model: str,
    cfg: InferenceConfig,
    seed: int,
    n_simulations: int | None = None,
    max_num_epochs: int | None = None,
    output_dir: str | Path | None = None,
) -> SNPEResult:
    torch, _, SNPE, posterior_nn, _ = _require_torch_and_sbi()
    torch.manual_seed(seed)
    n_sim = cfg.n_simulations_snpe if n_simulations is None else int(n_simulations)
    n_epochs = cfg.snpe_epochs if max_num_epochs is None else int(max_num_epochs)

    start = time.perf_counter()
    theta_np, x_np = _simulate_surviving_dataset(model, cfg, seed, n_sim)
    prior = _make_sbi_prior(model)
    embedding_net = _embedding_class()(input_size=x_np.shape[-1], embedding_dim=cfg.snpe_embedding_dim)
    density_estimator_builder = posterior_nn(model="nsf", embedding_net=embedding_net)
    inference = SNPE(prior=prior, density_estimator=density_estimator_builder)
    theta = torch.as_tensor(theta_np, dtype=torch.float32)
    x = torch.as_tensor(x_np, dtype=torch.float32)
    density_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=cfg.snpe_batch_size,
        validation_fraction=cfg.snpe_validation_fraction,
        max_num_epochs=n_epochs,
        show_train_summary=False,
    )
    posterior = inference.build_posterior(density_estimator)
    elapsed = time.perf_counter() - start

    training_losses, validation_losses = _extract_losses(density_estimator)
    curve_path = _plot_losses(
        training_losses,
        validation_losses,
        Path(output_dir or cfg.outputs_dir / f"snpe_{model}_loss.png"),
    )
    return SNPEResult(
        model=model,
        posterior=posterior,
        training_theta=theta_np,
        training_x=x_np,
        training_losses=training_losses,
        validation_losses=validation_losses,
        elapsed_seconds=elapsed,
        training_curve_path=curve_path,
    )


def sample_posterior(
    result: SNPEResult,
    observed_observables: np.ndarray,
    *,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    torch, _, _, _, _ = _require_torch_and_sbi()
    torch.manual_seed(seed)
    x = torch.as_tensor(np.asarray(observed_observables, dtype=np.float32)[None, ...])
    requested = int(num_samples)
    if result.model != "baseline":
        samples = result.posterior.sample((requested,), x=x, show_progress_bars=False)
        return np.asarray(samples.detach().cpu().numpy(), dtype=np.float64)

    collected: list[np.ndarray] = []
    remaining = requested
    attempt = 0
    max_attempts = 20
    while remaining > 0 and attempt < max_attempts:
        batch_size = max(remaining * 2, 256)
        samples = result.posterior.sample((batch_size,), x=x, show_progress_bars=False)
        batch = np.asarray(samples.detach().cpu().numpy(), dtype=np.float64)
        valid = batch[_valid_theta_mask(result.model, batch)]
        if valid.size:
            collected.append(valid[:remaining])
            remaining -= min(remaining, valid.shape[0])
        attempt += 1
    if remaining > 0:
        raise RuntimeError("SNPE posterior sampling failed to produce enough valid baseline draws.")
    return np.concatenate(collected, axis=0)[:requested]


def posterior_log_prob(
    result: SNPEResult,
    theta: np.ndarray,
    observed_observables: np.ndarray,
) -> float:
    torch, _, _, _, _ = _require_torch_and_sbi()
    theta_tensor = torch.as_tensor(np.asarray(theta, dtype=np.float32)[None, ...])
    x_tensor = torch.as_tensor(np.asarray(observed_observables, dtype=np.float32)[None, ...])
    value = result.posterior.log_prob(theta_tensor, x=x_tensor)
    return float(np.asarray(value.detach().cpu().numpy()).reshape(()))


def posterior_predictive(
    result: SNPEResult,
    posterior_samples: np.ndarray,
    observed_result: dict[str, Any],
    *,
    cfg: InferenceConfig,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    last = np.asarray(observed_result["trajectory"], dtype=np.float64)[-1]
    order = _parameter_order(result.model)
    predictive = []
    iterator = np.asarray(posterior_samples, dtype=np.float64)
    for idx, theta in enumerate(tqdm(iterator, desc=f"SNPE posterior predictive ({result.model})", leave=False)):
        theta_dict = {name: float(value) for name, value in zip(order, theta, strict=True)}
        sim = simulate_from_theta(
            theta_dict,
            model=result.model,
            cfg=cfg,
            seed=int(rng.integers(0, 2**31 - 1)) + idx,
            N0=float(last[0]),
            C0=float(last[5]),
            T=cfg.forecast_steps,
        )
        predictive.append(observable_matrix(sim["trajectory"]))
    return np.stack(predictive, axis=0)

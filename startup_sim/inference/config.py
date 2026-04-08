from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class InferenceConfig:
    """Top-level configuration for inference and evaluation."""

    dt: float = 1.0 / 12.0
    observation_steps: int = 60
    # The spec leaves H unspecified; default to one forecast year.
    forecast_steps: int = 12
    n_simulations_snpe: int = 50_000
    snpe_posterior_samples: int = 5_000
    n_evaluation_trials: int = 100
    snpe_embedding_dim: int = 32
    snpe_validation_fraction: float = 0.1
    snpe_epochs: int = 50
    snpe_batch_size: int = 256
    mcmc_num_warmup: int = 1_000
    mcmc_num_samples: int = 3_000
    mcmc_num_chains: int = 4
    mcmc_target_accept_prob: float = 0.8
    mcmc_thinning_for_ess: int = 1
    mcmc_platform: str = "cpu"
    mcmc_trial_workers: int = 0
    baseline_revenue_obs_rel_sigma: float = 0.01
    baseline_revenue_obs_min_sigma: float = 100.0
    baseline_burn_obs_rel_sigma: float = 0.01
    baseline_burn_obs_min_sigma: float = 100.0
    baseline_cash_obs_rel_sigma: float = 0.01
    baseline_cash_obs_min_sigma: float = 1_000.0
    sbc_posterior_samples: int = 500
    posterior_predictive_central_mass: float = 0.8
    outputs_dir: Path = field(default_factory=lambda: Path("outputs"))


DEFAULT_CONFIG = InferenceConfig()

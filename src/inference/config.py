from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class InferenceConfig:
    dt: float = 1.0 / 12.0
    obs_steps: int = 60
    pred_steps: int = 12
    snpe_sims: int = 50_000
    post_samples: int = 5_000
    trials: int = 100
    embed_dim: int = 32
    val_frac: float = 0.1
    epochs: int = 50
    batch_size: int = 256
    warmup: int = 1_000
    samples: int = 3_000
    chains: int = 4
    target_accept: float = 0.8
    thin: int = 1
    platform: str = "cpu"
    jobs: int = 0
    rev_rel_sigma: float = 0.01
    rev_min_sigma: float = 100.0
    burn_rel_sigma: float = 0.01
    burn_min_sigma: float = 100.0
    cash_rel_sigma: float = 0.01
    cash_min_sigma: float = 1_000.0
    sbc_samples: int = 500
    central_mass: float = 0.8
    out_dir: Path = field(default_factory=lambda: Path("outputs"))


DEFAULT_CONFIG = InferenceConfig()

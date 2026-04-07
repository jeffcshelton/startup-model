from __future__ import annotations

from typing import Any

import startup_sim.advanced as advanced_model
import startup_sim.baseline as baseline_model

DEFAULT_MODEL = baseline_model.MODEL_NAME
MODEL_NAMES = (baseline_model.MODEL_NAME, advanced_model.MODEL_NAME)
DEFAULT_PARAMS = dict(baseline_model.DEFAULT_PARAMS)

_MODELS = {
    baseline_model.MODEL_NAME: baseline_model,
    advanced_model.MODEL_NAME: advanced_model,
}


def normalize_model_name(model: str | None) -> str:
    """Normalize a user-provided model name."""

    candidate = DEFAULT_MODEL if model is None else model.strip().lower()
    if candidate not in _MODELS:
        valid_names = ", ".join(MODEL_NAMES)
        raise ValueError(f"unknown model {model!r}; expected one of: {valid_names}")
    return candidate


def get_model_module(model: str | None = None):
    """Return the simulation module for the requested model."""

    return _MODELS[normalize_model_name(model)]


def get_default_params(model: str | None = None) -> dict[str, Any]:
    """Return a copy of the default parameters for one model."""

    return dict(get_model_module(model).DEFAULT_PARAMS)


def simulate(*, model: str | None = None, **params: Any) -> dict[str, Any]:
    """Dispatch to either the baseline or advanced simulator."""

    module = get_model_module(model)
    return module.simulate(**params)


def batch_simulate(
    n_runs: int,
    base_params: dict[str, Any] | None = None,
    seed: int | None = None,
    *,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Dispatch to either the baseline or advanced batch simulator."""

    module = get_model_module(model)
    return module.batch_simulate(n_runs=n_runs, base_params=base_params, seed=seed)

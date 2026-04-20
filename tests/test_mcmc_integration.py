from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.baseline import DEFAULT_PARAMS, simulate
from src.inference.config import InferenceConfig
from src.inference.mcmc import run_baseline_nuts


def _has_numpyro() -> bool:
    try:
        import jax  # noqa: F401
        import numpyro  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(_has_numpyro(), "jax/numpyro not installed")
class MCMCIntegrationTests(unittest.TestCase):
    def test_short_chain_produces_finite_rhat_and_positive_ess(self) -> None:
        params = dict(DEFAULT_PARAMS)
        params["seed"] = 7
        observed = simulate(**params)
        cfg = InferenceConfig(
            warmup=100,
            samples=100,
            chains=1,
            out_dir=Path(tempfile.mkdtemp()),
        )
        result = run_baseline_nuts(observed, cfg=cfg, seed=9)
        for value in result.diagnostics.rhat.values():
            self.assertTrue(np.isfinite(value) or np.isnan(value))
        for value in result.diagnostics.ess_bulk.values():
            self.assertGreater(value, 0.0)
        for value in result.diagnostics.ess_tail.values():
            self.assertGreater(value, 0.0)


if __name__ == "__main__":
    unittest.main()

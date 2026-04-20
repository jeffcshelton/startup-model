from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from startup_sim.inference.config import InferenceConfig
from startup_sim.inference.snpe import train_snpe


def _has_snpe() -> bool:
    try:
        import torch  # noqa: F401
        import sbi  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(_has_snpe(), "torch/sbi not installed")
class SNPEIntegrationTests(unittest.TestCase):
    def test_reduced_training_has_finite_nonincreasing_loss_signal(self) -> None:
        cfg = InferenceConfig(
            n_simulations_snpe=500,
            snpe_epochs=5,
            outputs_dir=Path(tempfile.mkdtemp()),
        )
        result = train_snpe(model="baseline", cfg=cfg, seed=11, n_simulations=500, max_num_epochs=5)
        self.assertTrue(all(np.isfinite(result.training_losses)))
        if len(result.training_losses) >= 2:
            self.assertLessEqual(result.training_losses[-1], result.training_losses[0] + 1e-6)


if __name__ == "__main__":
    unittest.main()

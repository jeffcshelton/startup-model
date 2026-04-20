from __future__ import annotations

import unittest

import numpy as np

from startup_sim import baseline as baseline_model
from startup_sim.inference.likelihood import baseline_log_likelihood


class BaselineLikelihoodTests(unittest.TestCase):
    def test_returns_negative_infinity_when_cash_ruins_before_t(self) -> None:
        theta = {
            "p": 0.03,
            "q": 0.38,
            "K": 50_000.0,
            "v": 50.0,
            "gamma": 200.0,
            "b0": 2_000_000.0,
            "sigma_N": 5.0,
        }
        customers = np.array([10.0, 12.0, 15.0, 18.0], dtype=np.float64)
        value = baseline_log_likelihood(theta, customers, dt=1.0 / 12.0, initial_cash=100_000.0)
        self.assertTrue(np.isneginf(value))

    def test_full_trajectory_likelihood_penalizes_wrong_revenue_and_cash_history(self) -> None:
        truth = {
            "p": 0.03,
            "q": 0.38,
            "K": 50_000.0,
            "v": 100.0,
            "gamma": 40.0,
            "b0": 50_000.0,
            "sigma_N": 5.0,
            "N0": 10.0,
            "C0": 2_000_000.0,
            "T": 24,
            "dt": 1.0 / 12.0,
            "seed": 0,
        }
        observed = baseline_model.simulate(**truth)
        wrong = {
            "p": truth["p"],
            "q": truth["q"],
            "K": truth["K"],
            "v": 140.0,
            "gamma": 60.0,
            "b0": 35_000.0,
            "sigma_N": truth["sigma_N"],
        }
        ll_true = baseline_log_likelihood(
            truth,
            observed,
            dt=truth["dt"],
            revenue_obs_sigma=100.0,
            burn_obs_sigma=100.0,
            cash_obs_sigma=1_000.0,
        )
        ll_wrong = baseline_log_likelihood(
            wrong,
            observed,
            dt=truth["dt"],
            revenue_obs_sigma=100.0,
            burn_obs_sigma=100.0,
            cash_obs_sigma=1_000.0,
        )
        self.assertGreater(ll_true, ll_wrong)


if __name__ == "__main__":
    unittest.main()

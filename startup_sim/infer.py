from __future__ import annotations

import argparse

from startup_sim.inference.config import DEFAULT_CONFIG, InferenceConfig
from startup_sim.inference.evaluation import run_evaluation_study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run startup_sim inference and evaluation.")
    parser.add_argument("--method", choices=("mcmc", "snpe"), required=True)
    parser.add_argument("--model", choices=("baseline", "advanced"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=DEFAULT_CONFIG.n_evaluation_trials)
    parser.add_argument("--n-sim", dest="n_simulations_snpe", type=int, default=DEFAULT_CONFIG.n_simulations_snpe)
    parser.add_argument("--epochs", dest="snpe_epochs", type=int, default=DEFAULT_CONFIG.snpe_epochs)
    parser.add_argument("--forecast-steps", type=int, default=DEFAULT_CONFIG.forecast_steps)
    parser.add_argument("--observation-steps", type=int, default=DEFAULT_CONFIG.observation_steps)
    parser.add_argument("--mcmc-platform", choices=("cpu", "gpu"), default=DEFAULT_CONFIG.mcmc_platform)
    parser.add_argument("--jobs", dest="mcmc_trial_workers", type=int, default=DEFAULT_CONFIG.mcmc_trial_workers)
    parser.add_argument("--baseline-revenue-obs-rel-sigma", type=float, default=DEFAULT_CONFIG.baseline_revenue_obs_rel_sigma)
    parser.add_argument("--baseline-revenue-obs-min-sigma", type=float, default=DEFAULT_CONFIG.baseline_revenue_obs_min_sigma)
    parser.add_argument("--baseline-burn-obs-rel-sigma", type=float, default=DEFAULT_CONFIG.baseline_burn_obs_rel_sigma)
    parser.add_argument("--baseline-burn-obs-min-sigma", type=float, default=DEFAULT_CONFIG.baseline_burn_obs_min_sigma)
    parser.add_argument("--baseline-cash-obs-rel-sigma", type=float, default=DEFAULT_CONFIG.baseline_cash_obs_rel_sigma)
    parser.add_argument("--baseline-cash-obs-min-sigma", type=float, default=DEFAULT_CONFIG.baseline_cash_obs_min_sigma)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = InferenceConfig(
        observation_steps=args.observation_steps,
        forecast_steps=args.forecast_steps,
        n_evaluation_trials=args.trials,
        n_simulations_snpe=args.n_simulations_snpe,
        snpe_epochs=args.snpe_epochs,
        mcmc_platform=args.mcmc_platform,
        mcmc_trial_workers=args.mcmc_trial_workers,
        baseline_revenue_obs_rel_sigma=args.baseline_revenue_obs_rel_sigma,
        baseline_revenue_obs_min_sigma=args.baseline_revenue_obs_min_sigma,
        baseline_burn_obs_rel_sigma=args.baseline_burn_obs_rel_sigma,
        baseline_burn_obs_min_sigma=args.baseline_burn_obs_min_sigma,
        baseline_cash_obs_rel_sigma=args.baseline_cash_obs_rel_sigma,
        baseline_cash_obs_min_sigma=args.baseline_cash_obs_min_sigma,
    )
    summary = run_evaluation_study(method=args.method, model=args.model, cfg=cfg, seed=args.seed)
    print(summary.summary_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import replace

from src.inference.config import DEFAULT_CONFIG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run startup_sim inference and evaluation.")
    parser.add_argument("--method", choices=("mcmc", "snpe"), required=True)
    parser.add_argument("--model", choices=("baseline", "advanced"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--snpe-sims", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--pred-steps", type=int)
    parser.add_argument("--obs-steps", type=int)
    parser.add_argument("--platform", choices=("cpu", "gpu"))
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--rev-rel-sigma", type=float)
    parser.add_argument("--rev-min-sigma", type=float)
    parser.add_argument("--burn-rel-sigma", type=float)
    parser.add_argument("--burn-min-sigma", type=float)
    parser.add_argument("--cash-rel-sigma", type=float)
    parser.add_argument("--cash-min-sigma", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from src.inference.evaluation import run_evaluation_study

    overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in {"method", "model", "seed"} and value is not None
    }
    cfg = replace(DEFAULT_CONFIG, **overrides)
    summary = run_evaluation_study(method=args.method, model=args.model, cfg=cfg, seed=args.seed)
    print(summary.summary_path)


if __name__ == "__main__":
    main()

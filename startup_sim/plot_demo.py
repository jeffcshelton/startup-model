from __future__ import annotations

import argparse

from startup_sim.model import DEFAULT_MODEL, MODEL_NAMES, get_default_params, normalize_model_name, simulate
from startup_sim.plotting import plot_with_matplotlib, plot_with_plotly


def _add_baseline_arguments(parser: argparse.ArgumentParser, defaults: dict[str, float | int]) -> None:
    """Attach baseline model arguments."""

    parser.add_argument("--p", type=float, default=float(defaults["p"]))
    parser.add_argument("--q", type=float, default=float(defaults["q"]))
    parser.add_argument("--K", type=float, default=float(defaults["K"]))
    parser.add_argument("--v", type=float, default=float(defaults["v"]))
    parser.add_argument("--gamma", type=float, default=float(defaults["gamma"]))
    parser.add_argument("--b0", type=float, default=float(defaults["b0"]))
    parser.add_argument("--sigma-n", dest="sigma_N", type=float, default=float(defaults["sigma_N"]))
    parser.add_argument("--N0", type=float, default=float(defaults["N0"]))
    parser.add_argument("--C0", type=float, default=float(defaults["C0"]))
    parser.add_argument("--T", type=int, default=int(defaults["T"]))
    parser.add_argument("--dt", type=float, default=float(defaults["dt"]))


def _add_advanced_arguments(parser: argparse.ArgumentParser, defaults: dict[str, float | int]) -> None:
    """Attach advanced model arguments."""

    parser.add_argument("--p", type=float, default=float(defaults["p"]))
    parser.add_argument("--q", type=float, default=float(defaults["q"]))
    parser.add_argument("--kappa", type=float, default=float(defaults["kappa"]))
    parser.add_argument("--sigma-q", dest="sigma_q", type=float, default=float(defaults["sigma_q"]))
    parser.add_argument("--K", type=float, default=float(defaults["K"]))
    parser.add_argument("--v", type=float, default=float(defaults["v"]))
    parser.add_argument("--epsilon", type=float, default=float(defaults["epsilon"]))
    parser.add_argument("--chi", type=float, default=float(defaults["chi"]))
    parser.add_argument("--gamma", type=float, default=float(defaults["gamma"]))
    parser.add_argument("--b0", type=float, default=float(defaults["b0"]))
    parser.add_argument("--alpha", type=float, default=float(defaults["alpha"]))
    parser.add_argument("--sigma-n", dest="sigma_N", type=float, default=float(defaults["sigma_N"]))
    parser.add_argument("--N0", type=float, default=float(defaults["N0"]))
    parser.add_argument("--C0", type=float, default=float(defaults["C0"]))
    parser.add_argument("--T", type=int, default=int(defaults["T"]))
    parser.add_argument("--dt", type=float, default=float(defaults["dt"]))


def main() -> None:
    """Run the simulator and show a visualization."""

    root_parser = argparse.ArgumentParser(description="Visualize startup_sim output.")
    root_parser.add_argument("--engine", choices=("matplotlib", "plotly"), default="matplotlib")
    root_parser.add_argument("--model", choices=MODEL_NAMES, default=DEFAULT_MODEL)
    root_parser.add_argument("--seed", type=int, default=7)
    root_args, remaining = root_parser.parse_known_args()

    model = normalize_model_name(root_args.model)
    defaults = get_default_params(model)
    parser = argparse.ArgumentParser(description="Visualize startup_sim output.")
    parser.add_argument("--engine", choices=("matplotlib", "plotly"), default=root_args.engine)
    parser.add_argument("--model", choices=MODEL_NAMES, default=model)
    parser.add_argument("--seed", type=int, default=root_args.seed)
    if model == "baseline":
        _add_baseline_arguments(parser, defaults)
    else:
        _add_advanced_arguments(parser, defaults)

    args = parser.parse_args(remaining, namespace=root_args)
    result = simulate(model=model, **{key: value for key, value in vars(args).items() if key not in {"engine", "model"}})
    title = f"startup_sim ({model})"
    if args.engine == "matplotlib":
        plot_with_matplotlib(result, title=title)
    else:
        plot_with_plotly(result, title=title)


if __name__ == "__main__":
    main()

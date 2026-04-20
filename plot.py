from __future__ import annotations

import argparse

from src.model import DEFAULT_MODEL, MODEL_NAMES, get_default_params, normalize_model_name, simulate
from src.visualization import plot_with_matplotlib


def main() -> None:
    root_parser = argparse.ArgumentParser(description="Plot startup simulator output.")
    root_parser.add_argument("--interactive", action="store_true")
    root_parser.add_argument("--model", choices=MODEL_NAMES, default=DEFAULT_MODEL)
    root_parser.add_argument("--seed", type=int, default=7)
    root_parser.add_argument("--host", default="127.0.0.1")
    root_parser.add_argument("--port", type=int, default=8050)
    root_args, remaining = root_parser.parse_known_args()

    model = normalize_model_name(root_args.model)
    if root_args.interactive:
        from src.interactive_plot import launch_interactive_explorer

        parser = argparse.ArgumentParser(description="Plot startup simulator output.")
        parser.add_argument("--interactive", action="store_true")
        parser.add_argument("--model", choices=MODEL_NAMES, default=DEFAULT_MODEL)
        parser.add_argument("--seed", type=int, default=7)
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8050)
        args = parser.parse_args()
        launch_interactive_explorer(model=args.model, host=args.host, port=args.port)
        return

    defaults = get_default_params(model)
    parser = argparse.ArgumentParser(description="Plot startup simulator output.")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--model", choices=MODEL_NAMES, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    if model == "baseline":
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
    else:
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

    args = parser.parse_args(remaining, namespace=root_args)
    result = simulate(
        model=model,
        **{
            key: value
            for key, value in vars(args).items()
            if key not in {"interactive", "model", "host", "port"}
        },
    )
    plot_with_matplotlib(result, title=f"startup_sim ({model})")


if __name__ == "__main__":
    main()

from __future__ import annotations

from src.model import DEFAULT_MODEL, MODEL_NAMES, get_default_params, normalize_model_name, simulate
import argparse

def main():
    root_parser = argparse.ArgumentParser(description="Run the startup growth simulator.")
    root_parser.add_argument("--model", choices=MODEL_NAMES, default=DEFAULT_MODEL)
    root_parser.add_argument("--seed", type=int, default=None)
    root_args, remaining = root_parser.parse_known_args()
    model = normalize_model_name(root_args.model)
    defaults = get_default_params(model)

    parser = argparse.ArgumentParser(description="Run the startup growth simulator.")
    parser.add_argument("--model", choices=MODEL_NAMES, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=None)
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
    result = simulate(model=model, **{key: value for key, value in vars(args).items() if key != "model"})
    last_row = result["trajectory"][-1]
    summary = (
        "startup_sim OK "
        f"model={model} "
        f"customers={last_row[0]:.3f} "
        f"cash={last_row[5]:.3f} "
        f"revenue={last_row[3]:.3f} "
        f"burn={last_row[4]:.3f} "
    )
    if model == "advanced":
        summary += f"virality={last_row[6]:.3f} "
    summary += f"survived={result['survived']} ruin_time={result['ruin_time']}"
    print(summary)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from startup_sim.model import DEFAULT_PARAMS, simulate


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Parser for simulation arguments.
    """

    parser = argparse.ArgumentParser(description="Run the startup growth simulator.")
    parser.add_argument("--p", type=float, default=float(DEFAULT_PARAMS["p"]))
    parser.add_argument("--q", type=float, default=float(DEFAULT_PARAMS["q"]))
    parser.add_argument("--K", type=float, default=float(DEFAULT_PARAMS["K"]))
    parser.add_argument("--v", type=float, default=float(DEFAULT_PARAMS["v"]))
    parser.add_argument("--gamma", type=float, default=float(DEFAULT_PARAMS["gamma"]))
    parser.add_argument("--b0", type=float, default=float(DEFAULT_PARAMS["b0"]))
    parser.add_argument("--sigma-n", dest="sigma_N", type=float, default=float(DEFAULT_PARAMS["sigma_N"]))
    parser.add_argument("--N0", type=float, default=float(DEFAULT_PARAMS["N0"]))
    parser.add_argument("--C0", type=float, default=float(DEFAULT_PARAMS["C0"]))
    parser.add_argument("--T", type=int, default=int(DEFAULT_PARAMS["T"]))
    parser.add_argument("--dt", type=float, default=float(DEFAULT_PARAMS["dt"]))
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    """Run the simulator from the command line and print ending stats.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Prints a concise summary of the simulation outcome.
    """

    args = build_parser().parse_args()
    result = simulate(
        p=args.p,
        q=args.q,
        K=args.K,
        v=args.v,
        gamma=args.gamma,
        b0=args.b0,
        sigma_N=args.sigma_N,
        N0=args.N0,
        C0=args.C0,
        T=args.T,
        dt=args.dt,
        seed=args.seed,
    )
    last_row = result["trajectory"][-1]
    print(
        "startup_sim OK "
        f"customers={last_row[0]:.3f} "
        f"cash={last_row[5]:.3f} "
        f"revenue={last_row[3]:.3f} "
        f"burn={last_row[4]:.3f} "
        f"survived={result['survived']} "
        f"ruin_time={result['ruin_time']}"
    )


if __name__ == "__main__":
    main()

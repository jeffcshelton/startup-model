from __future__ import annotations

import argparse

from startup_sim.model import DEFAULT_PARAMS, simulate
from startup_sim.plotting import plot_with_matplotlib


def main() -> None:
    """Run the simulator and show a visualization.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Displays the requested visualization.
    """

    parser = argparse.ArgumentParser(description="Visualize startup_sim output with matplotlib.")
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
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

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
    plot_with_matplotlib(result, title="startup_sim")


if __name__ == "__main__":
    main()

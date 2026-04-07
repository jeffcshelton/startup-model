from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go

import startup_sim.advanced as advanced_model
import startup_sim.baseline as baseline_model
from startup_sim.model import DEFAULT_MODEL, MODEL_NAMES, get_default_params, normalize_model_name
from startup_sim.plotting import build_plotly_figure


@dataclass(frozen=True)
class ControlSpec:
    """Slider metadata for the interactive explorer."""

    name: str
    label: str
    minimum: float
    maximum: float
    value: float
    step: float
    cast: Callable[[float], Any]


def _make_control_specs(model: str) -> list[ControlSpec]:
    """Build slider metadata for the requested model."""

    defaults = get_default_params(model)
    common = [
        ControlSpec("p", "p", -1.0, 1.0, float(defaults["p"]), 0.01, float),
        ControlSpec("q", "q", -1.0, 1.0, float(defaults["q"]), 0.01, float),
        ControlSpec("K", "K", 5_000.0, 100_000.0, float(defaults["K"]), 500.0, float),
        ControlSpec("v", "v", 10.0, 250.0, float(defaults["v"]), 1.0, float),
        ControlSpec("gamma", "gamma", 0.0, 200.0, float(defaults["gamma"]), 1.0, float),
        ControlSpec("b0", "b0", 0.0, 250_000.0, float(defaults["b0"]), 1_000.0, float),
        ControlSpec("sigma_N", "sigma_N", 0.0, 50.0, float(defaults["sigma_N"]), 0.5, float),
        ControlSpec("N0", "N0", 0.0, 10_000.0, float(defaults["N0"]), 10.0, float),
        ControlSpec("C0", "C0", 0.0, 10_000_000.0, float(defaults["C0"]), 50_000.0, float),
        ControlSpec("T", "T", 12.0, 240.0, float(defaults["T"]), 1.0, int),
        ControlSpec("dt", "dt", 1.0 / 24.0, 0.5, float(defaults["dt"]), 1.0 / 120.0, float),
    ]
    if model == "advanced":
        common[2:2] = [
            ControlSpec("kappa", "kappa", 0.0, 5.0, float(defaults["kappa"]), 0.05, float),
            ControlSpec("sigma_q", "sigma_q", 0.0, 0.5, float(defaults["sigma_q"]), 0.005, float),
        ]
        common[5:5] = [
            ControlSpec("epsilon", "epsilon", 0.0, 0.95, float(defaults["epsilon"]), 0.01, float),
            ControlSpec("chi", "chi", 0.0, 2.0, float(defaults["chi"]), 0.01, float),
        ]
        common.insert(9, ControlSpec("alpha", "alpha", 0.0, 1_000.0, float(defaults["alpha"]), 10.0, float))

    common.extend(
        [
            ControlSpec("iterations", "iterations", 1.0, 100.0, 20.0, 1.0, int),
            ControlSpec("seed", "seed", 0.0, 10_000.0, 7.0, 1.0, int),
        ]
    )
    return common


def _format_mark_value(value: float) -> str:
    """Format a compact slider mark label."""

    if abs(value) >= 1_000_000.0:
        return f"{value / 1_000_000.0:.1f}M"
    if abs(value) >= 1_000.0:
        return f"{value / 1_000.0:.0f}K"
    if abs(value) >= 10.0 or value.is_integer():
        return f"{value:.0f}"
    if abs(value) >= 1.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _slider_marks(spec: ControlSpec) -> dict[float, str]:
    """Build sparse slider marks."""

    mark_values = np.linspace(spec.minimum, spec.maximum, num=5, dtype=np.float64)
    values = sorted({float(spec.minimum), float(spec.value), float(spec.maximum), *mark_values.tolist()})
    return {value: _format_mark_value(value) for value in values}


def _control_card(spec: ControlSpec) -> html.Div:
    """Build a control card for the Dash layout."""

    return html.Div(
        [
            html.Div(spec.label, className="control-label"),
            dcc.Slider(
                id=f"control-{spec.name}",
                min=spec.minimum,
                max=spec.maximum,
                step=spec.step,
                value=spec.value,
                marks=_slider_marks(spec),
                included=False,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ],
        className="control-card",
    )


def _params_from_controls(model: str, controls: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Map UI controls into simulator parameters."""

    if model == "baseline":
        v = float(controls["v"])
        gamma = float(controls["gamma"])
        note: str | None = None
        if gamma >= v:
            note = "gamma >= v makes the baseline model invalid and will raise an error."
        return (
            {
                "p": float(controls["p"]),
                "q": float(controls["q"]),
                "K": float(controls["K"]),
                "v": v,
                "gamma": gamma,
                "b0": float(controls["b0"]),
                "sigma_N": float(controls["sigma_N"]),
                "N0": float(controls["N0"]),
                "C0": float(controls["C0"]),
                "T": int(controls["T"]),
                "dt": float(controls["dt"]),
            },
            note,
        )

    return (
        {
            "p": float(controls["p"]),
            "q": float(controls["q"]),
            "kappa": float(controls["kappa"]),
            "sigma_q": float(controls["sigma_q"]),
            "K": float(controls["K"]),
            "v": float(controls["v"]),
            "epsilon": float(controls["epsilon"]),
            "chi": float(controls["chi"]),
            "gamma": float(controls["gamma"]),
            "b0": float(controls["b0"]),
            "alpha": float(controls["alpha"]),
            "sigma_N": float(controls["sigma_N"]),
            "N0": float(controls["N0"]),
            "C0": float(controls["C0"]),
            "T": int(controls["T"]),
            "dt": float(controls["dt"]),
        },
        None,
    )


def build_app(model: str = DEFAULT_MODEL) -> Dash:
    """Build the Dash application."""

    model = normalize_model_name(model)
    control_specs = _make_control_specs(model)
    simulator = baseline_model if model == "baseline" else advanced_model

    app = Dash(__name__)
    app.title = f"startup_sim explorer ({model})"
    app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    margin: 0;
                    background: linear-gradient(135deg, #f7f4ee 0%, #efe7d7 100%);
                    color: #1f2937;
                    font-family: Georgia, 'Times New Roman', serif;
                }
                .page {
                    display: grid;
                    grid-template-columns: 360px minmax(0, 1fr);
                    gap: 24px;
                    padding: 24px;
                }
                .sidebar, .plot-shell {
                    background: rgba(255, 253, 248, 0.92);
                    border: 1px solid rgba(191, 167, 129, 0.35);
                    border-radius: 24px;
                    box-shadow: 0 18px 60px rgba(89, 63, 28, 0.12);
                }
                .sidebar {
                    padding: 22px 20px 24px 20px;
                    position: sticky;
                    top: 24px;
                    align-self: start;
                }
                .plot-shell {
                    padding: 12px;
                }
                .eyebrow {
                    text-transform: uppercase;
                    letter-spacing: 0.18em;
                    font-size: 11px;
                    color: #9a6b2f;
                    margin-bottom: 8px;
                }
                .headline {
                    font-size: 30px;
                    line-height: 1.1;
                    margin: 0 0 10px 0;
                }
                .subhead {
                    font-size: 14px;
                    line-height: 1.5;
                    color: #475569;
                    margin: 0 0 20px 0;
                }
                .control-grid {
                    display: grid;
                    grid-template-columns: 1fr;
                    gap: 12px;
                }
                .control-card {
                    padding: 12px 14px;
                    border-radius: 16px;
                    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,242,232,0.98));
                    border: 1px solid rgba(210, 190, 160, 0.65);
                }
                .control-label {
                    font-size: 12px;
                    font-weight: 700;
                    letter-spacing: 0.08em;
                    text-transform: uppercase;
                    margin-bottom: 10px;
                    color: #7c5a2f;
                }
                .status {
                    margin-top: 18px;
                    padding: 12px 14px;
                    border-radius: 14px;
                    background: #f3ebdf;
                    color: #4b5563;
                    font-size: 13px;
                    line-height: 1.5;
                    border: 1px solid rgba(180, 155, 120, 0.4);
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    """
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(f"startup_sim {model}", className="eyebrow"),
                            html.H1("distribution explorer", className="headline"),
                            html.P(
                                "Adjust the sliders to rerun the simulator. "
                                "Each update overlays multiple stochastic runs "
                                "with reduced opacity so the distribution stays visible.",
                                className="subhead",
                            ),
                            html.Div([_control_card(spec) for spec in control_specs], className="control-grid"),
                            html.Div(id="status", className="status"),
                        ],
                        className="sidebar",
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                id="figure",
                                config={"displaylogo": False, "scrollZoom": True},
                                style={"height": "calc(100vh - 48px)"},
                            )
                        ],
                        className="plot-shell",
                    ),
                ],
                className="page",
            ),
        ]
    )

    @app.callback(
        Output("figure", "figure"),
        Output("status", "children"),
        [Input(f"control-{spec.name}", "value") for spec in control_specs],
    )
    def _update_figure(*slider_values: float) -> tuple[go.Figure, str]:
        """Recompute the simulation distribution when controls change."""

        controls = {
            spec.name: spec.cast(float(value))
            for spec, value in zip(control_specs, slider_values, strict=True)
        }
        params, note = _params_from_controls(model, controls)
        iterations = max(int(controls["iterations"]), 1)
        seed = int(controls["seed"])
        try:
            results = simulator.batch_simulate(iterations, base_params=params, seed=seed)
        except ValueError as exc:
            return go.Figure(), f"{model} run failed: {exc}"

        status = f"{iterations} {model} runs rendered with seed root {seed}."
        if note is not None:
            status = f"{status} {note}"
        return build_plotly_figure(results, title=f"startup_sim distribution explorer ({model})", note=note), status

    return app


def launch_interactive_explorer(
    model: str = DEFAULT_MODEL,
    host: str = "127.0.0.1",
    port: int = 8050,
) -> None:
    """Launch the Dash explorer."""

    app = build_app(model=model)
    run_method = getattr(app, "run", None)
    if run_method is not None:
        run_method(host=host, port=port, debug=False)
    else:
        app.run_server(host=host, port=port, debug=False)


def main() -> None:
    """Launch the explorer from the command line."""

    parser = argparse.ArgumentParser(description="Interactive explorer for startup_sim.")
    parser.add_argument("--model", choices=MODEL_NAMES, default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    launch_interactive_explorer(model=args.model, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

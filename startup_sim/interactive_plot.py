from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from dash import Dash, Input, Output, dcc, html

from startup_sim.model import DEFAULT_PARAMS, batch_simulate
from startup_sim.plotting import build_plotly_figure


@dataclass(frozen=True)
class ControlSpec:
    """Slider metadata for the interactive explorer.

    Parameters
    ----------
    name : str
        Parameter name.
    label : str
        User-facing label.
    minimum : float
        Minimum slider value.
    maximum : float
        Maximum slider value.
    value : float
        Initial slider value.
    step : float
        Slider step size.
    cast : callable
        Function used to cast the raw slider value.
    """

    name: str
    label: str
    minimum: float
    maximum: float
    value: float
    step: float
    cast: Callable[[float], Any]


CONTROL_SPECS: list[ControlSpec] = [
    ControlSpec("p", "p", -1.0, 1.0, float(DEFAULT_PARAMS["p"]), 0.01, float),
    ControlSpec("q", "q", -1.0, 1.0, float(DEFAULT_PARAMS["q"]), 0.01, float),
    ControlSpec("K", "K", 5_000.0, 100_000.0, float(DEFAULT_PARAMS["K"]), 500.0, float),
    ControlSpec("v", "v", 10.0, 250.0, float(DEFAULT_PARAMS["v"]), 1.0, float),
    ControlSpec("gamma", "gamma", 0.0, 200.0, float(DEFAULT_PARAMS["gamma"]), 1.0, float),
    ControlSpec("b0", "b0", 0.0, 250_000.0, float(DEFAULT_PARAMS["b0"]), 1_000.0, float),
    ControlSpec("sigma_N", "sigma_N", 0.0, 50.0, float(DEFAULT_PARAMS["sigma_N"]), 0.5, float),
    ControlSpec("N0", "N0", 0.0, 10_000.0, float(DEFAULT_PARAMS["N0"]), 10.0, float),
    ControlSpec("C0", "C0", 0.0, 10_000_000.0, float(DEFAULT_PARAMS["C0"]), 50_000.0, float),
    ControlSpec("T", "T", 12.0, 240.0, float(DEFAULT_PARAMS["T"]), 1.0, int),
    ControlSpec("dt", "dt", 1.0 / 24.0, 0.5, float(DEFAULT_PARAMS["dt"]), 1.0 / 120.0, float),
    ControlSpec("iterations", "iterations", 1.0, 100.0, 20.0, 1.0, int),
    ControlSpec("seed", "seed", 0.0, 10_000.0, 7.0, 1.0, int),
]


def _format_mark_value(value: float) -> str:
    """Format a compact slider mark label.

    Parameters
    ----------
    value : float
        Mark value.

    Returns
    -------
    str
        Compact label text.
    """

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
    """Build sparse slider marks.

    Parameters
    ----------
    spec : ControlSpec
        Slider metadata.

    Returns
    -------
    dict of float to str
        Slider marks.
    """

    mark_values = np.linspace(spec.minimum, spec.maximum, num=5, dtype=np.float64)
    values = sorted({float(spec.minimum), float(spec.value), float(spec.maximum), *mark_values.tolist()})
    return {value: _format_mark_value(value) for value in values}


def _control_card(spec: ControlSpec) -> html.Div:
    """Build a control card for the Dash layout.

    Parameters
    ----------
    spec : ControlSpec
        Slider metadata.

    Returns
    -------
    dash.html.Div
        Dash layout node.
    """

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


def _params_from_controls(controls: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Map UI controls into simulator parameters.

    Parameters
    ----------
    controls : dict of str to Any
        Slider values.

    Returns
    -------
    tuple of dict of str to Any and str or None
        Simulator parameters and an optional note.
    """

    v = float(controls["v"])
    gamma = float(controls["gamma"])
    delta = max(v - gamma, 1.0e-6)
    note: str | None = None
    if gamma >= v:
        note = "gamma >= v, so net unit margin was clipped to a small positive value."

    return (
        {
            "p": float(controls["p"]),
            "q": float(controls["q"]),
            "K": float(controls["K"]),
            "v": v,
            "gamma": float(gamma),
            "b0": float(controls["b0"]),
            "sigma_N": float(controls["sigma_N"]),
            "N0": float(controls["N0"]),
            "C0": float(controls["C0"]),
            "T": int(controls["T"]),
            "dt": float(controls["dt"]),
        },
        note,
    )


def build_app() -> Dash:
    """Build the Dash application.

    Parameters
    ----------
    None

    Returns
    -------
    dash.Dash
        Configured Dash application.
    """

    app = Dash(__name__)
    app.title = "startup_sim explorer"
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
                            html.Div("startup_sim", className="eyebrow"),
                            html.H1("distribution explorer", className="headline"),
                            html.P(
                                "Adjust the sliders to rerun the simulator. "
                                "Each update overlays multiple stochastic runs "
                                "with reduced opacity so the distribution stays visible.",
                                className="subhead",
                            ),
                            html.Div([_control_card(spec) for spec in CONTROL_SPECS], className="control-grid"),
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
        [Input(f"control-{spec.name}", "value") for spec in CONTROL_SPECS],
    )
    def _update_figure(*slider_values: float) -> tuple[go.Figure, str]:
        """Recompute the simulation distribution when controls change.

        Parameters
        ----------
        *slider_values : float
            Current slider values.

        Returns
        -------
        tuple of plotly.graph_objects.Figure and str
            Updated figure and status string.
        """

        controls = {
            spec.name: spec.cast(float(value))
            for spec, value in zip(CONTROL_SPECS, slider_values, strict=True)
        }
        params, note = _params_from_controls(controls)
        iterations = max(int(controls["iterations"]), 1)
        seed = int(controls["seed"])
        results = batch_simulate(iterations, base_params=params, seed=seed)
        status = f"{iterations} runs rendered with seed root {seed}."
        if note is not None:
            status = f"{status} {note}"
        return build_plotly_figure(results, title="startup_sim distribution explorer", note=note), status

    return app


def launch_interactive_explorer(host: str = "127.0.0.1", port: int = 8050) -> None:
    """Launch the Dash explorer.

    Parameters
    ----------
    host : str, default="127.0.0.1"
        Host interface for the Dash server.
    port : int, default=8050
        Port for the Dash server.

    Returns
    -------
    None
        Starts the server.
    """

    app = build_app()
    run_method = getattr(app, "run", None)
    if run_method is not None:
        run_method(host=host, port=port, debug=False)
    else:
        app.run_server(host=host, port=port, debug=False)


def main() -> None:
    """Launch the explorer from the command line.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Starts the server.
    """

    parser = argparse.ArgumentParser(description="Interactive explorer for startup_sim.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    launch_interactive_explorer(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

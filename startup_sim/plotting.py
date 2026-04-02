from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SERIES_COLORS: dict[str, str] = {
    "customers": "#0f766e",
    "cash": "#0b3c5d",
    "monthly revenue": "#d97706",
    "acquired": "#059669",
    "lost": "#dc2626",
}


def _money_scale(values: np.ndarray) -> tuple[float, str]:
    """Choose a display scale for money-valued series.

    Parameters
    ----------
    values : numpy.ndarray
        Values to scale.

    Returns
    -------
    tuple of float and str
        Scale factor and unit label.
    """

    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    if max_abs >= 1_000_000.0:
        return 1_000_000.0, "$M"
    if max_abs >= 1_000.0:
        return 1_000.0, "$K"
    return 1.0, "$"


def _realized_flows(customers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute realized customer gains and losses.

    Parameters
    ----------
    customers : numpy.ndarray
        Customer trajectory.

    Returns
    -------
    tuple of numpy.ndarray and numpy.ndarray
        Acquired and lost customer counts.
    """

    delta_customers = np.diff(customers, prepend=customers[0])
    return np.maximum(delta_customers, 0.0), np.maximum(-delta_customers, 0.0)


def _panel_data(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build panel data shared by matplotlib and Plotly rendering.

    Parameters
    ----------
    results : list of dict of str to Any
        Simulation results from the same model.

    Returns
    -------
    list of dict of str to Any
        Plot panel specifications.
    """

    raw_runs = [np.asarray(result["trajectory"], dtype=np.float64) for result in results]
    dt = float(results[0]["params"]["dt"])
    market_size = float(results[0]["params"]["K"])

    customer_runs = [run[:, 0] for run in raw_runs]
    cash_runs_raw = [run[:, 5] for run in raw_runs]
    revenue_runs_raw = [run[:, 3] * dt for run in raw_runs]
    acquired_runs: list[np.ndarray] = []
    lost_runs: list[np.ndarray] = []
    for customers in customer_runs:
        acquired, lost = _realized_flows(customers)
        acquired_runs.append(acquired)
        lost_runs.append(lost)

    cash_scale, cash_unit = _money_scale(np.concatenate(cash_runs_raw))
    revenue_scale, revenue_unit = _money_scale(np.concatenate(revenue_runs_raw))

    return [
        {
            "title": "customers",
            "unit": "",
            "series": [{"label": "customers", "runs": customer_runs}],
            "reference": ("TAM", market_size),
        },
        {
            "title": "cash",
            "unit": cash_unit,
            "series": [{"label": "cash", "runs": [series / cash_scale for series in cash_runs_raw]}],
            "reference": ("zero cash", 0.0),
        },
        {
            "title": "monthly revenue",
            "unit": revenue_unit,
            "series": [{"label": "monthly revenue", "runs": [series / revenue_scale for series in revenue_runs_raw]}],
            "reference": None,
        },
        {
            "title": "monthly customer flows",
            "unit": "",
            "series": [
                {"label": "acquired", "runs": acquired_runs},
                {"label": "lost", "runs": lost_runs},
            ],
            "reference": None,
        },
    ]


def _line_alpha(n_runs: int) -> float:
    """Choose an overlay alpha value.

    Parameters
    ----------
    n_runs : int
        Number of runs being overlaid.

    Returns
    -------
    float
        Alpha channel value.
    """

    if n_runs <= 1:
        return 0.85
    return float(min(0.45, max(0.04, 1.2 / np.sqrt(n_runs))))


def build_plotly_figure(
    results: list[dict[str, Any]],
    title: str | None = None,
    note: str | None = None,
) -> go.Figure:
    """Build a Plotly figure from one or more simulation runs.

    Parameters
    ----------
    results : list of dict of str to Any
        Simulation outputs to visualize.
    title : str or None, default=None
        Optional figure title.
    note : str or None, default=None
        Optional subtitle note.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure with all runs overlaid.
    """

    if not results:
        raise ValueError("results must not be empty.")

    time_axis = np.arange(len(results[0]["trajectory"]), dtype=np.float64) * float(results[0]["params"]["dt"])
    panels = _panel_data(results)
    figure = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[panel["title"] for panel in panels],
    )
    alpha = _line_alpha(len(results))

    for row_index, panel in enumerate(panels, start=1):
        for series in panel["series"]:
            color = SERIES_COLORS[series["label"]]
            rgba = _hex_to_rgba(color, alpha)
            mean_rgba = _hex_to_rgba(color, 0.95)
            for values in series["runs"]:
                figure.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=values,
                        mode="lines",
                        line={"color": rgba, "width": 1.0},
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=series["label"],
                        name=series["label"],
                    ),
                    row=row_index,
                    col=1,
                )

            mean_values = np.mean(np.stack(series["runs"], axis=0), axis=0)
            figure.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=mean_values,
                    mode="lines",
                    line={"color": mean_rgba, "width": 3.0},
                    name=series["label"],
                    legendgroup=series["label"],
                    showlegend=True,
                ),
                row=row_index,
                col=1,
            )

        if panel["reference"] is not None:
            label, value = panel["reference"]
            axis_values = np.concatenate([np.concatenate(series["runs"]) for series in panel["series"]])
            if float(np.min(axis_values)) <= value <= float(np.max(axis_values)):
                figure.add_hline(
                    y=value,
                    row=row_index,
                    col=1,
                    line_dash="dash",
                    line_color="#dc2626",
                    line_width=1.5,
                    annotation_text=label,
                    annotation_position="top left",
                )

        yaxis_title = panel["title"] if not panel["unit"] else f"{panel['title']} ({panel['unit']})"
        figure.update_yaxes(title_text=yaxis_title, row=row_index, col=1)

    title_text = "startup_sim"
    if title is not None:
        title_text = title
    if note is not None:
        title_text = f"{title_text}<br><sup>{note}</sup>"

    max_time = float(time_axis[-1]) if time_axis.size else 0.0
    dtick = 0.25 if max_time <= 2.0 else 0.5 if max_time <= 6.0 else 1.0

    figure.update_layout(
        template="plotly_white",
        height=360 * len(panels),
        margin={"l": 80, "r": 40, "t": 110, "b": 70},
        title={"text": title_text, "x": 0.03, "xanchor": "left"},
        paper_bgcolor="#f7f4ee",
        plot_bgcolor="#fffdf8",
        font={"family": "Georgia, Times New Roman, serif", "size": 13, "color": "#1f2937"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "bgcolor": "rgba(255,255,255,0.75)",
        },
        hovermode="x unified",
    )
    figure.update_xaxes(
        title_text="Time (years)",
        showgrid=True,
        gridcolor="#e7dfd2",
        showticklabels=True,
        tickmode="linear",
        tick0=0.0,
        dtick=dtick,
        ticks="outside",
    )
    figure.update_yaxes(showgrid=True, gridcolor="#efe8dc", zerolinecolor="#d4c3b0")
    figure.update_annotations(font={"size": 16, "color": "#6b4f2d"})
    return figure

def plot_with_matplotlib(result: dict[str, Any], title: str | None = None) -> None:
    """Display a matplotlib visualization for one run.

    Parameters
    ----------
    result : dict of str to Any
        Simulation output.
    title : str or None, default=None
        Optional figure title.

    Returns
    -------
    None
        Opens the matplotlib figure.
    """

    panels = _panel_data([result])
    time_axis = np.arange(len(result["trajectory"]), dtype=np.float64) * float(result["params"]["dt"])
    figure, axes = plt.subplots(len(panels), 1, figsize=(10.0, 3.0 * len(panels)), sharex=True)
    axes_array = np.atleast_1d(axes)

    for axis, panel in zip(axes_array, panels):
        for series in panel["series"]:
            axis.plot(time_axis, series["runs"][0], linewidth=2.0, label=series["label"], color=SERIES_COLORS[series["label"]])

        if panel["reference"] is not None:
            label, value = panel["reference"]
            y_values = np.concatenate([series["runs"][0] for series in panel["series"]])
            if float(np.min(y_values)) <= value <= float(np.max(y_values)):
                axis.axhline(value, color="#dc2626", linestyle="--", linewidth=1.5, label=label)

        yaxis_title = panel["title"] if not panel["unit"] else f"{panel['title']} ({panel['unit']})"
        axis.set_ylabel(yaxis_title)
        axis.legend(loc="best")
        axis.grid(alpha=0.25)

    axes_array[-1].set_xlabel("Time (years)")
    if title is not None:
        figure.suptitle(title)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    else:
        figure.tight_layout()
    plt.show()


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color to an rgba string.

    Parameters
    ----------
    hex_color : str
        Color in ``#RRGGBB`` form.
    alpha : float
        Alpha channel value.

    Returns
    -------
    str
        CSS-style rgba string.
    """

    stripped = hex_color.lstrip("#")
    red = int(stripped[0:2], 16)
    green = int(stripped[2:4], 16)
    blue = int(stripped[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha:.4f})"

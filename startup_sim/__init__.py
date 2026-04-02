from startup_sim.model import DEFAULT_PARAMS, batch_simulate, simulate

__all__ = [
    "DEFAULT_PARAMS",
    "simulate",
    "batch_simulate",
    "plot_with_matplotlib",
    "build_plotly_figure",
    "launch_interactive_explorer",
]


def plot_with_matplotlib(*args, **kwargs):
    """Dispatch to the matplotlib visualizer.

    Parameters
    ----------
    *args : Any
        Positional arguments forwarded to the visualizer.
    **kwargs : Any
        Keyword arguments forwarded to the visualizer.

    Returns
    -------
    None
        Displays the matplotlib figure.
    """

    from startup_sim.plotting import plot_with_matplotlib as _plot_with_matplotlib

    return _plot_with_matplotlib(*args, **kwargs)


def build_plotly_figure(*args, **kwargs):
    """Dispatch to the Plotly figure builder.

    Parameters
    ----------
    *args : Any
        Positional arguments forwarded to the builder.
    **kwargs : Any
        Keyword arguments forwarded to the builder.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure.
    """

    from startup_sim.plotting import build_plotly_figure as _build_plotly_figure

    return _build_plotly_figure(*args, **kwargs)


def launch_interactive_explorer(*args, **kwargs):
    """Dispatch to the interactive explorer.

    Parameters
    ----------
    *args : Any
        Positional arguments forwarded to the explorer.
    **kwargs : Any
        Keyword arguments forwarded to the explorer.

    Returns
    -------
    None
        Starts the explorer server.
    """

    from startup_sim.interactive_plot import (
        launch_interactive_explorer as _launch_interactive_explorer,
    )

    return _launch_interactive_explorer(*args, **kwargs)

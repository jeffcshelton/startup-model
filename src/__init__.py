from .advanced import (
    DEFAULT_PARAMS as ADVANCED_DEFAULT_PARAMS,
)
from .advanced import (
    batch_simulate as batch_simulate_advanced,
)
from .advanced import (
    simulate as simulate_advanced,
)
from .baseline import (
    DEFAULT_PARAMS as BASELINE_DEFAULT_PARAMS,
)
from .baseline import (
    batch_simulate as batch_simulate_baseline,
)
from .baseline import (
    simulate as simulate_baseline,
)
from .model import (
    DEFAULT_MODEL,
    DEFAULT_PARAMS,
    MODEL_NAMES,
    batch_simulate,
    get_default_params,
    get_model_module,
    normalize_model_name,
    simulate,
)
from .inference import DEFAULT_CONFIG as DEFAULT_INFERENCE_CONFIG
from .inference import InferenceConfig, baseline_log_likelihood, baseline_survival_indicator

__all__ = [
    "ADVANCED_DEFAULT_PARAMS",
    "BASELINE_DEFAULT_PARAMS",
    "DEFAULT_MODEL",
    "DEFAULT_PARAMS",
    "MODEL_NAMES",
    "batch_simulate",
    "batch_simulate_advanced",
    "batch_simulate_baseline",
    "get_default_params",
    "get_model_module",
    "normalize_model_name",
    "simulate",
    "simulate_advanced",
    "simulate_baseline",
    "DEFAULT_INFERENCE_CONFIG",
    "InferenceConfig",
    "baseline_log_likelihood",
    "baseline_survival_indicator",
    "plot_with_matplotlib",
    "plot_with_plotly",
    "build_plotly_figure",
    "launch_interactive_explorer",
]


def plot_with_matplotlib(*args, **kwargs):
    """Dispatch to the matplotlib visualizer."""

    from .visualization import plot_with_matplotlib as _plot_with_matplotlib

    return _plot_with_matplotlib(*args, **kwargs)


def plot_with_plotly(*args, **kwargs):
    """Dispatch to the Plotly visualizer."""

    from .visualization import plot_with_plotly as _plot_with_plotly

    return _plot_with_plotly(*args, **kwargs)


def build_plotly_figure(*args, **kwargs):
    """Dispatch to the Plotly figure builder."""

    from .visualization import build_plotly_figure as _build_plotly_figure

    return _build_plotly_figure(*args, **kwargs)


def launch_interactive_explorer(*args, **kwargs):
    """Dispatch to the interactive explorer."""

    from .interactive_plot import (
        launch_interactive_explorer as _launch_interactive_explorer,
    )

    return _launch_interactive_explorer(*args, **kwargs)

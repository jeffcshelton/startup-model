# startup_sim

`startup_sim` simulates startup customer growth under stochastic Bass-style adoption with cash dynamics and ruin detection.

## Requirements

- Python 3.10+
- `numpy`
- `scipy`

## Command Line

Run the simulator and print ending summary statistics:

```bash
python -m startup_sim --p 0.03 --q 0.38 --K 50000 --v 100 --gamma 40 --b0 50000 --sigma-n 5 --N0 10 --C0 2000000 --T 60
```

## Python API

```python
from startup_sim import simulate, DEFAULT_PARAMS

result = simulate(**DEFAULT_PARAMS, seed=7)
print(result["trajectory"][-1])
```

## Visualization

Static views are available through matplotlib:

```bash
python -m startup_sim.plot_demo
```

An interactive Plotly/Dash explorer is available here:

```bash
python -m startup_sim.interactive_plot
```

Additional visualization requirements:

- `matplotlib`
- `plotly`
- `dash`

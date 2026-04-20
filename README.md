# startup_sim

`startup_sim` simulates startup customer growth, cash dynamics, and ruin detection with two distinct models:

- `baseline`: the original stochastic Bass-style adoption model from `main`
- `advanced`: the expanded checkpoint2 model with latent virality, churn, and price compression

## Requirements

- Python 3.10+
- `numpy`
- `scipy`
- `jax`
- `numpyro`
- `torch`
- `sbi`
- `matplotlib`

## Nix Shells

The repository now exposes two shells:

```bash
nix develop
```

For the simulator, plotting, and general development.

```bash
nix develop .#inference
```

For the inference stack, including the custom `sbi` and `numpyro` derivations.

## Command Line

Run the baseline simulator and print ending summary statistics:

```bash
python -m startup_sim --model baseline --p 0.03 --q 0.38 --K 50000 --v 100 --gamma 40 --b0 50000 --sigma-n 5 --N0 10 --C0 2000000 --T 60
```

Run the advanced simulator:

```bash
python -m startup_sim --model advanced --p 0.03 --q 0.38 --kappa 1.0 --sigma-q 0.05 --K 50000 --v 100 --epsilon 0.1 --chi 0.02 --gamma 40 --b0 50000 --alpha 200 --sigma-n 5 --N0 10 --C0 2000000 --T 60
```

## Python API

```python
from startup_sim import (
    BASELINE_DEFAULT_PARAMS,
    ADVANCED_DEFAULT_PARAMS,
    simulate,
    simulate_advanced,
    simulate_baseline,
)

baseline_result = simulate_baseline(**BASELINE_DEFAULT_PARAMS, seed=7)
advanced_result = simulate_advanced(**ADVANCED_DEFAULT_PARAMS, seed=7)
same_dispatch = simulate(model="baseline", **BASELINE_DEFAULT_PARAMS, seed=7)
print(baseline_result["trajectory"][-1])
print(advanced_result["trajectory"][-1])
print(same_dispatch["model"])
```

## Visualization

Static views are available through both matplotlib and Plotly:

```bash
python -m startup_sim.plot_demo --model baseline --engine matplotlib
python -m startup_sim.plot_demo --model advanced --engine plotly
```

An interactive Plotly/Dash explorer is available for either model:

```bash
python -m startup_sim.interactive_plot --model baseline
python -m startup_sim.interactive_plot --model advanced
```

Additional visualization requirements:

- `matplotlib`
- `plotly`
- `dash`

## Inference

Run baseline NUTS evaluation:

```bash
python -m startup_sim.infer --method mcmc --model baseline --seed 0
```

Run SNPE evaluation on either model:

```bash
python -m startup_sim.infer --method snpe --model baseline --seed 0
python -m startup_sim.infer --method snpe --model advanced --seed 0
```

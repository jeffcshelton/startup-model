# Startup Simulator

This startup simulator models customer growth, cash dynamics, and run detection
for hypothetical startup companies with given metrics. It consists of a
simulator, which plays forward a startup's trajectory given certain parameters,
as well as methods of inferring the original parameters given noisy,
data-limited trajectory paths using uncertainty quantification.

There are two models implemented, with corresponding UQ methods tailored to
each.

### Baseline

This is a Bass-style adoption model with ruin detection.

### Advanced

Expansion on the Bass model that add latent virality, churn, and price
compression.

**TODO:** Full descriptions of models.

## Usage

### Requirements

The simulators and UQ methods are built with Python, and they require the
following packages to be installed to function:

- Python 3.10+
- `numpy`
- `scipy`
- `jax`
- `numpyro`
- `torch`
- `sbi`
- `matplotlib`
- `plotly`
- `dash`

A Nix development shell is provided to fully instantiate the environment with
all relevant tools and libraries. It can be accessed with:

```bash
nix develop
```

### Simulator

```bash
python sim.py --model baseline --p 0.03 --q 0.38 --K 50000 --v 100 --gamma 40 --b0 50000 --sigma-n 5 --N0 10 --C0 2000000 --T 60
```

```bash
python sim.py --model advanced --p 0.03 --q 0.38 --kappa 1.0 --sigma-q 0.05 --K 50000 --v 100 --epsilon 0.1 --chi 0.02 --gamma 40 --b0 50000 --alpha 200 --sigma-n 5 --N0 10 --C0 2000000 --T 60
```

### Inference

```bash
python uq.py --method mcmc --model baseline --seed 0
```

```bash
python uq.py --method snpe --model baseline --seed 0
python uq.py --method snpe --model advanced --seed 0
```

### Visualization

```bash
python plot.py --model baseline
python plot.py --model advanced
```

```bash
python plot.py --interactive --model baseline
python plot.py --interactive --model advanced
```

## AI Usage

The core models, design decisions, full testing and validation, literature
review, and all key implementation were completed by humans.

ChatGPT and Claude were used to aid the implementation of the visualization
tools, including the plotting and interactive web interface. These are
UI-intensive tasks related to the presentation, not the performance, of the
models.

AI models were additionally used for research of different model techniques with
efficiency comparisons and light editing of the repository: primarily inline
documentation generation (with human review), formatting, unit tests to
double-check the model(s) logic (again, with review), and API shaping for
integration with the visualization code.

No additions from AI tools have been made without author approval and
understanding.

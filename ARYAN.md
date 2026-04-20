# Repository Walk-Through

This repository has two main responsibilities:

1. simulate startup growth and cash dynamics under two related models
2. run uncertainty quantification and posterior inference on top of those simulators

At a high level, the codebase is split into:

- `startup_sim/`: the simulator package, plotting, and CLI entry points
- `startup_sim/inference/`: likelihoods, MCMC, SNPE, evaluation metrics, and experiment harnesses
- `tests/`: unit and reduced integration tests
- `outputs/`: saved diagnostic plots and JSON summaries from inference runs
- `flake.nix`: Nix environments for simulator-only work and inference work

The simulator part is small and fairly clean. The UQ part is where most of the engineering complexity lives.

## 1. Core Conceptual Model

The repository models a startup as a dynamical system with:

- customer count
- revenue
- burn
- cash
- optionally a latent virality state

There are two simulator variants:

- `baseline`: Bass-style customer adoption with stochastic customer noise and simple linear cash economics
- `advanced`: a richer model with latent time-varying virality, churn, price compression, CAC-like growth cost, and deterministic integration between noise updates

The inference stack is deliberately asymmetric:

- baseline model: supports both closed-form approximate likelihood + NUTS MCMC, and likelihood-free SNPE
- advanced model: supports SNPE only

That asymmetry is intentional. The baseline model has a tractable approximate transition likelihood for customers, while the advanced model does not.

## 2. Package Surface

The public package re-exports in `startup_sim/__init__.py` are meant to give one import surface for:

- default parameter dictionaries
- direct baseline and advanced simulators
- model-dispatching helpers
- plotting helpers
- the main inference config and baseline likelihood helpers

This means the repo can be used either as a CLI project or as a Python library.

## 3. Simulator Dispatch Layer

`startup_sim/model.py` is the dispatch layer.

Its job is simple:

- define the recognized model names
- normalize user input like `"baseline"` or `"advanced"`
- return the correct module
- expose a `simulate(...)` wrapper that forwards to the chosen model
- expose `batch_simulate(...)` the same way

This file contains almost no model logic. It exists to keep downstream CLIs and plotting code model-agnostic.

## 4. Baseline Simulator

`startup_sim/baseline.py` implements the simpler startup model.

### 4.1 Parameters

The baseline parameters are:

- `p`, `q`, `K`: Bass diffusion / market adoption parameters
- `v`: revenue per customer
- `gamma`: marginal burn per customer
- `b0`: fixed baseline burn
- `sigma_N`: customer noise scale
- `N0`, `C0`: initial customers and cash
- `T`, `dt`: horizon and time step

Important support restriction:

- `v > gamma`

That means each customer must contribute positive unit economics before fixed burn. The simulator raises a `ValueError` if this is violated.

### 4.2 Dynamics

The customer drift is:

- `growth_drift(customers, p, q, K)`

which is the Bass-style deterministic adoption component:

- growth is stronger when the market is unsaturated
- growth is amplified by word-of-mouth through `q * customers / K`

The state update is Euler-Maruyama:

- customers evolve with drift plus Gaussian noise
- cash evolves deterministically given the current customers

Specifically:

- `next_customers = clip(customers + drift * dt + noise, 0, K)`
- `next_cash = cash + ((v - gamma) * customers - b0) * dt`

Customer noise scales like:

- `sigma_N * sqrt(customers * dt) * Normal(0, 1)`

### 4.3 Recorded Trajectory

Each row of the baseline trajectory has 6 columns:

1. `customers`
2. `acquired`
3. `0.0`
4. `revenue`
5. `burn`
6. `cash`

The third column is always zero in the baseline model because explicit churn is not modeled there.

### 4.4 Ruin Handling

If cash crosses zero, the run is marked ruined:

- `ruin_time` is recorded
- the remaining rows are filled by repeating the ruin state

That convention matters later for posterior predictive checks and failure-mode classification.

## 5. Advanced Simulator

`startup_sim/advanced.py` is the richer model.

This file adds three major mechanisms that do not exist in the baseline model:

- a latent virality state `q(t)`
- explicit churn `chi * customers`
- more realistic economics through price compression and customer acquisition cost coupling

### 5.1 Extra Parameters

Compared with the baseline model, the advanced model introduces:

- `kappa`: mean reversion rate for latent virality
- `sigma_q`: diffusion scale for latent virality
- `epsilon`: price compression with market saturation
- `chi`: churn rate
- `alpha`: CAC-like growth cost proportional to positive net customer growth

### 5.2 Customer and Cash Dynamics

The advanced model separates:

- gross acquisition flow
- net customer drift after churn
- revenue with saturation-driven compression
- cash drift including CAC pressure

Key functions:

- `acquisition_flow(...)`
- `customer_drift(...)`
- `revenue(...)`
- `cash_drift(...)`

Economically:

- revenue per customer falls as saturation increases through `epsilon`
- burn increases with customers through `gamma`
- additional cost is charged when net growth is positive through `alpha * max(net_growth, 0)`

### 5.3 Numerical Method

The advanced model is hybrid:

- deterministic state evolution for `(customers, cash)` is integrated over each step with `scipy.integrate.solve_ivp`
- after each deterministic step, stochastic updates are applied to latent virality `q` and to customers

This is a more complicated numerical scheme than the baseline Euler-Maruyama update, and it is one reason exact likelihood-based inference is not attempted here.

### 5.4 Latent Virality

The latent `q` is an OU-like mean-reverting process:

- it reverts toward the nominal `q`
- it receives Gaussian perturbations scaled by `sigma_q`
- it is clipped below by `Q_MIN`

That latent state is recorded in the last trajectory column.

### 5.5 Recorded Trajectory

Each advanced trajectory row has 7 columns:

1. `customers`
2. `acquired`
3. `churned`
4. `revenue`
5. `burn`
6. `cash`
7. latent `q`

## 6. Visualization Layer

The simulator visualization code lives in:

- `startup_sim/plotting.py`
- `startup_sim/plot_demo.py`
- `startup_sim/interactive_plot.py`

### 6.1 Shared Plot Builder

`startup_sim/plotting.py` converts trajectories into a panelized view.

It computes:

- customers
- cash
- monthly revenue
- monthly realized gains/losses in customers
- latent `q` for the advanced model

The plotting code supports:

- static matplotlib rendering
- interactive Plotly rendering

This code is presentation-only. It does not affect simulation or inference.

### 6.2 Demo CLI

`startup_sim/plot_demo.py` is a simple “simulate once and plot it” command.

It:

- parses model-specific parameters
- runs one simulation
- renders it with either matplotlib or Plotly

### 6.3 Dash Explorer

`startup_sim/interactive_plot.py` builds a Dash application with sliders for parameters.

This is essentially a manual model explorer:

- choose model
- move sliders
- rerun simulation
- inspect resulting trajectory overlays

It is useful for intuition, not for inference.

## 7. Main Simulator CLI

`startup_sim/__main__.py` is the main simulator entry point.

Running `python -m startup_sim ...`:

- selects a model
- attaches model-specific CLI arguments
- runs one simulation
- prints final-state summary statistics

This is the most direct way to sanity-check the simulator from the shell.

## 8. Inference Package Overview

All UQ code lives in `startup_sim/inference/`.

The files are organized by responsibility:

- `config.py`: experiment-level hyperparameters
- `utils.py`: parameter ordering, priors, observable extraction, simulator wrappers
- `likelihood.py`: baseline closed-form likelihood and survival logic
- `mcmc.py`: NumPyro NUTS for the baseline model
- `snpe.py`: `sbi`-based sequential neural posterior estimation
- `metrics.py`: calibration, sharpness, and scoring metrics
- `plotting.py`: SBC/PIT/posterior predictive plots for inference results
- `evaluation.py`: full simulation-study harness

## 9. Shared Inference Utilities

`startup_sim/inference/utils.py` is the glue layer between simulators and inference.

### 9.1 Parameter Orders

It defines canonical parameter orders:

- `BASELINE_PARAM_ORDER`
- `ADVANCED_PARAM_ORDER`

That matters because:

- posterior samples are stored as arrays
- rank histograms and KDEs need consistent parameter ordering
- SNPE and MCMC need a stable mapping between vectors and named parameters

### 9.2 Observable Matrix

`observable_matrix(...)` converts a raw simulator trajectory into the 5 observable dimensions used by inference and evaluation:

1. `cash`
2. `revenue`
3. `customers`
4. `delta_n_plus`
5. `delta_n_minus`

These are the quantities the SNPE encoder sees and the evaluation harness scores.

### 9.3 Prior Sampler

`prior_sampler(model)` returns a vectorized NumPy sampler for the specified model prior.

This is used by:

- trial generation in the evaluation harness
- SNPE simulation dataset generation
- prior variance estimation for the identification metric

### 9.4 Simulator Wrappers

`simulate_from_theta(...)` and `split_observed_and_future(...)` are convenience wrappers that:

- take a parameter dictionary
- call the right simulator
- enforce shared config values like `dt`
- optionally continue a run into a future forecast horizon

`sample_surviving_trial(...)` repeatedly draws from the prior until it gets an observed trajectory that survives through the observation window.

That conditioning is important because both the baseline MCMC likelihood and the SNPE training protocol condition on survival through the observed horizon.

## 10. Baseline Likelihood

`startup_sim/inference/likelihood.py` contains the baseline closed-form approximate likelihood.

This is one of the most important files in the repository.

### 10.1 What the Likelihood Conditions On

The current baseline likelihood uses:

- the observed customer history
- the observed revenue history
- the observed burn history
- the observed cash history

The original closed-form term is for the customer trajectory only. Revenue, burn, and cash are added through observation factors.

### 10.2 Customer Transition Likelihood

The baseline customer process is approximated by the one-step Gaussian Euler-Maruyama transition:

- mean = previous customers + Bass drift * `dt`
- variance = `sigma_N^2 * previous_customers * dt`

`baseline_log_likelihood(...)` applies that term across the full observed customer path.

### 10.3 Survival Hard Wall

The file reconstructs the implied cash path using:

- observed customer history
- proposed `v`, `gamma`, `b0`
- initial cash

This is done in `baseline_cash_path(...)`.

Then:

- `baseline_survival_indicator(...)` returns `0.0` if the cash path stays positive
- returns `-inf` otherwise

That is the “hard wall” corresponding to ruin.

### 10.4 Revenue, Burn, and Cash Observation Terms

The simulator makes revenue, burn, and cash deterministic once customers and parameters are fixed. Conditioning on exact equality would create a degenerate manifold that is awkward for HMC, so the repository uses small Gaussian observation models instead:

- revenue is centered at `v * customers`
- burn is centered at `b0 + gamma * customers`
- cash is centered at the reconstructed cash path

These extra terms are what make the baseline model “see” the full past time series, not just customers.

## 11. Baseline MCMC

`startup_sim/inference/mcmc.py` implements baseline posterior sampling with NumPyro NUTS.

### 11.1 Dependency Boundary

The file lazily imports JAX/NumPyro through `_require_numpyro()`.

That keeps the package importable even if the inference environment is not installed.

### 11.2 Numerics and Parameterization

Several details here are deliberate:

- JAX is set to `x64`
- the baseline support constraint `v > gamma` is enforced structurally, not by sampling invalid `v`
- the NUTS model uses a large finite negative wall (`NEGATIVE_HARD_WALL`) internally rather than literal `-inf` factors for compatibility with the installed NumPyro version

The sampler uses:

- `gamma ~ LogNormal(...)`
- `v_margin_log ~ Normal(...)`
- `v = gamma + exp(v_margin_log)`

plus a prior-correction factor so the effective prior on `v` still matches the intended LogNormal prior.

### 11.3 Model Terms

The NumPyro model contains:

- priors for baseline parameters
- the customer closed-form log-likelihood
- a hard survival constraint via `numpyro.factor`
- Gaussian observation likelihoods for revenue, burn, and cash

### 11.4 Initialization

`_baseline_init_values(...)` tries to construct a valid and reasonable initial state by reading the observed histories:

- revenue helps initialize `v`
- burn helps initialize `gamma` and `b0`
- cash can be used as a fallback

It then checks whether the proposed initial parameter set yields finite baseline likelihood.

### 11.5 Diagnostics

The file also implements:

- manual R-hat
- bulk ESS
- tail ESS
- trace plot generation

`compute_mcmc_diagnostics(...)` packages those values into a structured result.

### 11.6 Posterior Predictive

`posterior_predictive_baseline(...)` converts posterior samples back into parameter dictionaries and forward-simulates the baseline model from the observed terminal state:

- initial customers = observed `N_T`
- initial cash = observed `C_T`

The outputs are converted to the same 5-dimensional observable format used by evaluation.

## 12. SNPE

`startup_sim/inference/snpe.py` implements likelihood-free posterior estimation with `sbi`.

This file serves both the baseline and advanced models.

### 12.1 Why SNPE Exists Here

The advanced model is too complicated for the closed-form MCMC approach because it includes:

- a latent virality path
- nonlinear coupling in cash dynamics
- a path-dependent ruin event

SNPE avoids explicit likelihood evaluation by learning `q(theta | y)` from simulation.

### 12.2 Priors

The file defines model-specific prior components and passes them through `sbi`’s `process_prior(...)`.

This compatibility path exists because the installed `sbi` version expects Torch distributions rather than a custom joint prior wrapper.

### 12.3 Encoder

The current embedding network is an LSTM:

- input shape: `T x 5`
- output: a learned fixed-dimensional trajectory embedding

That embedding conditions a neural spline flow posterior via `posterior_nn(model="nsf", embedding_net=...)`.

### 12.4 Training Data Generation

`_simulate_surviving_dataset(...)` repeatedly:

- samples parameters from the prior
- simulates the chosen model
- discards failed runs during the observation window
- stores only surviving `(theta, observable_matrix(trajectory))` pairs

This matches the survival conditioning used elsewhere in the project.

### 12.5 Training

`train_snpe(...)`:

- simulates a training set
- builds the prior and embedding network
- trains `SNPE`
- extracts training and validation loss curves
- saves a loss plot

The result object stores:

- posterior object
- training samples
- losses
- elapsed time
- saved plot path

### 12.6 Sampling and Support Filtering

`sample_posterior(...)` draws posterior samples for one observed trajectory.

There is one extra fix specific to the baseline model:

- baseline posterior samples are filtered to enforce `v > gamma`

This is necessary because SNPE can place mass slightly outside the simulator’s valid support, even if training data came only from valid simulations.

### 12.7 Posterior Predictive

`posterior_predictive(...)` takes posterior samples, starts from the last observed state, and forward-simulates the chosen model into the forecast horizon.

This is the SNPE analogue of the MCMC posterior predictive rollout.

## 13. Evaluation Metrics

`startup_sim/inference/metrics.py` implements the metrics used in the simulation study.

### 13.1 Probabilistic Accuracy

- `energy_score_ensemble(...)`
- `energy_score_by_dimension(...)`

These compare predictive trajectory ensembles against the true future trajectory.

### 13.2 Parameter Accuracy

- `posterior_nll_snpe(...)`
- `posterior_nll_mcmc_kde(...)`

For MCMC, posterior density at the true parameter is approximated via KDE over posterior draws.

### 13.3 Calibration

- `sbc_rank(...)`
- `pit_values(...)`

SBC looks at parameter calibration. PIT looks at predictive calibration for observables over time.

### 13.4 Sharpness

- `prediction_interval_width(...)`
- `prediction_interval_coverage(...)`

These summarize how wide predictive intervals are and whether they cover the truth at the nominal rate.

### 13.5 Identification

- `posterior_std(...)`
- `posterior_variance_ratio(...)`

The variance ratio is posterior variance divided by prior variance. Smaller values mean stronger identification.

### 13.6 Failure Mode Characterization

- `classify_regime(...)`
- `ruin_probability(...)`

The regime classifier is heuristic:

- surviving
- slow-bleed ruin
- growth-induced ruin

This is not inferred by a latent classifier. It is post hoc categorization of simulated ground-truth outcomes.

## 14. Inference Plotting

`startup_sim/inference/plotting.py` saves:

- SBC rank histograms
- PIT histograms
- posterior predictive overlays

These are written to disk by the evaluation harness and are the main visual diagnostics for calibration.

## 15. Evaluation Harness

`startup_sim/inference/evaluation.py` is the top-level experiment driver.

This is the file that turns all the pieces into a simulation study.

### 15.1 Trial Structure

For one trial:

1. sample `theta_true` from the prior, conditioned on observed survival
2. simulate an observed window
3. simulate a future continuation from the same true parameters
4. run the requested inference method on the observed window
5. evaluate parameter and predictive performance against the true parameters and future rollout

### 15.2 Separate Paths for SNPE and MCMC

There are two single-trial functions:

- `_run_single_trial_with_snpe(...)`
- `_run_single_trial_with_mcmc(...)`

The MCMC path is baseline-only.

The SNPE path works for both models.

### 15.3 Aggregation

`run_evaluation_study(...)`:

- optionally trains one amortized SNPE posterior first
- runs `M` independent trials
- aggregates mean energy scores, NLL, posterior standard deviations, variance ratios, coverage, width, runtime, simulator calls, and failure-mode summaries
- writes `summary.json`
- writes SBC and PIT plots

### 15.4 Parallelism

For baseline MCMC evaluation only, trials can be parallelized with `ProcessPoolExecutor`.

That matters because:

- each trial is independent
- each trial already uses multiple chains internally

So the repository scales across CPU cores by parallelizing across independent trials, not by trying to over-parallelize one small NUTS job.

## 16. Inference Configuration

`startup_sim/inference/config.py` defines a single dataclass with the top-level experiment knobs:

- simulator time step and horizons
- SNPE simulation budget and epochs
- posterior sample counts
- MCMC warmup, draws, and chains
- MCMC backend platform
- worker count for MCMC study parallelism
- baseline observation noise scales for revenue, burn, and cash
- plotting/output settings

This keeps the experiment configuration centralized instead of scattering constants across files.

## 17. Inference CLI

`startup_sim/infer.py` is the inference/evaluation CLI.

This is the main experiment entry point.

It parses:

- inference method: `mcmc` or `snpe`
- model: `baseline` or `advanced`
- seed
- number of trials
- SNPE simulation budget and epochs
- forecast and observation horizons
- MCMC backend and worker count
- baseline observation-noise hyperparameters

It then builds an `InferenceConfig`, runs `run_evaluation_study(...)`, and prints the path to the generated `summary.json`.

## 18. Tests

The test suite is small but targeted.

### 18.1 Likelihood Unit Tests

`tests/test_likelihood.py` checks:

- ruin implies `-inf` log-likelihood
- wrong revenue/burn/cash histories are penalized relative to the true generating parameters

This is the most critical correctness check for the baseline likelihood.

### 18.2 MCMC Integration Test

`tests/test_mcmc_integration.py` runs a reduced baseline MCMC job and checks:

- R-hat values are finite or `NaN` in the one-chain case
- ESS values are positive

### 18.3 SNPE Integration Test

`tests/test_snpe_integration.py` runs a reduced SNPE training job and checks:

- training losses are finite
- the final loss does not increase relative to the first

These are smoke tests, not exhaustive statistical validation.

## 19. Outputs

The `outputs/` directory is where experiment artifacts go.

Typical contents are:

- `summary.json`
- `sbc_hist.png`
- `pit_hist.png`
- MCMC trace plots
- posterior predictive overlay plots if explicitly saved

This repository has existing saved outputs from prior runs. Those are results, not source-of-truth code. If results look surprising, the code in `startup_sim/inference/` is the place to inspect.

## 20. Nix Environments

`flake.nix` defines two shells:

- default shell: simulator, plotting, general development
- `.#inference`: heavier stack with JAX, NumPyro, Torch, and custom `sbi` derivations

The inference shell exists because:

- `sbi` is not available directly in nixpkgs here
- the inference dependencies need version pinning and custom overrides

This is why many inference validation commands should be run under:

```bash
nix develop .#inference
```

## 21. Typical Workflows

### 21.1 Run One Simulator

```bash
python -m startup_sim --model baseline
python -m startup_sim --model advanced
```

### 21.2 Visualize One Simulator Run

```bash
python -m startup_sim.plot_demo --model baseline --engine matplotlib
python -m startup_sim.plot_demo --model advanced --engine plotly
```

### 21.3 Launch Interactive Explorer

```bash
python -m startup_sim.interactive_plot --model baseline
python -m startup_sim.interactive_plot --model advanced
```

### 21.4 Run Baseline MCMC Evaluation

```bash
nix develop .#inference
python -m startup_sim.infer --method mcmc --model baseline --seed 0
```

### 21.5 Run SNPE Evaluation

```bash
nix develop .#inference
python -m startup_sim.infer --method snpe --model baseline --seed 0
python -m startup_sim.infer --method snpe --model advanced --seed 0
```

### 21.6 Fast Smoke Tests

```bash
nix develop .#inference
python -m unittest tests.test_likelihood -v
python -m unittest tests.test_mcmc_integration -v
python -m unittest tests.test_snpe_integration -v
```

## 22. How the Pieces Fit Together

The cleanest mental model for the repository is:

- `baseline.py` and `advanced.py` define the world
- `model.py` dispatches between those worlds
- `plotting.py` and friends help humans inspect trajectories from those worlds
- `likelihood.py` encodes what can be scored exactly for the baseline world
- `mcmc.py` samples the baseline posterior using that score
- `snpe.py` learns an amortized posterior for either world from simulation alone
- `metrics.py` defines what “good inference” means
- `evaluation.py` turns all of that into a reproducible simulation study

If you are new to the repo, the shortest path to understanding it is:

1. read `startup_sim/baseline.py`
2. read `startup_sim/advanced.py`
3. read `startup_sim/inference/utils.py`
4. read `startup_sim/inference/likelihood.py`
5. read `startup_sim/inference/mcmc.py`
6. read `startup_sim/inference/snpe.py`
7. read `startup_sim/inference/evaluation.py`

That sequence matches the dependency structure and will make the rest of the repository much easier to follow.

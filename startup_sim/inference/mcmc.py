from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from startup_sim.inference.config import InferenceConfig
from startup_sim.inference.likelihood import baseline_log_likelihood
from startup_sim.inference.utils import (
    BASELINE_PARAM_ORDER,
    baseline_theta_to_dict,
    observable_matrix,
    simulate_from_theta,
)

NEGATIVE_HARD_WALL = -1.0e30


def _require_numpyro():
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jax import random
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS, init_to_value
    except Exception as exc:  # pragma: no cover - exercised only when deps exist
        raise ImportError("NumPyro inference requires jax and numpyro to be installed.") from exc
    return jnp, random, numpyro, dist, MCMC, NUTS, init_to_value


@dataclass(slots=True)
class MCMCDiagnostics:
    rhat: dict[str, float]
    ess_bulk: dict[str, float]
    ess_tail: dict[str, float]
    non_converged: dict[str, bool]
    low_ess: dict[str, bool]
    survival_rejection_fraction: float


@dataclass(slots=True)
class MCMCResult:
    posterior_samples: dict[str, np.ndarray]
    diagnostics: MCMCDiagnostics
    elapsed_seconds: float
    trace_plot_paths: list[Path]
    posterior_predictive_observables: np.ndarray | None = None


def _autocorrelation(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    if np.allclose(x, 0.0):
        return np.ones(1, dtype=np.float64)
    fft = np.fft.rfft(np.concatenate([x, np.zeros_like(x)]))
    acov = np.fft.irfft(fft * np.conj(fft))[: len(x)]
    acov /= np.arange(len(x), 0, -1)
    acf = acov / acov[0]
    return np.clip(acf.real, -1.0, 1.0)


def _effective_sample_size(chains: np.ndarray) -> tuple[float, float]:
    m, n = chains.shape
    chain_means = np.mean(chains, axis=1)
    B = n * np.var(chain_means, ddof=1) if m > 1 else 0.0
    W = np.mean(np.var(chains, axis=1, ddof=1))
    var_hat = ((n - 1) / n) * W + B / n if n > 1 else W
    merged = chains.reshape(-1)
    acf = _autocorrelation(merged)
    tau = 1.0
    for k in range(1, len(acf) - 1, 2):
        pair_sum = acf[k] + acf[k + 1]
        if pair_sum <= 0.0:
            break
        tau += 2.0 * pair_sum
    ess_bulk = float((m * n) / max(tau, 1e-12))
    tail_indicator = (merged <= np.quantile(merged, 0.1)).astype(np.float64)
    tail_acf = _autocorrelation(tail_indicator)
    tau_tail = 1.0
    for k in range(1, len(tail_acf) - 1, 2):
        pair_sum = tail_acf[k] + tail_acf[k + 1]
        if pair_sum <= 0.0:
            break
        tau_tail += 2.0 * pair_sum
    ess_tail = float((m * n) / max(tau_tail, 1e-12))
    _ = var_hat
    return ess_bulk, ess_tail


def _rhat(chains: np.ndarray) -> float:
    m, n = chains.shape
    if n < 2:
        return float("nan")
    chain_means = np.mean(chains, axis=1)
    B = n * np.var(chain_means, ddof=1) if m > 1 else 0.0
    W = np.mean(np.var(chains, axis=1, ddof=1))
    if W <= 0.0:
        return 1.0
    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W))


def compute_mcmc_diagnostics(samples: dict[str, np.ndarray], thinning: int = 1) -> MCMCDiagnostics:
    rhat: dict[str, float] = {}
    ess_bulk: dict[str, float] = {}
    ess_tail: dict[str, float] = {}
    non_converged: dict[str, bool] = {}
    low_ess: dict[str, bool] = {}
    for name, values in samples.items():
        chains = np.asarray(values, dtype=np.float64)
        if chains.ndim != 2:
            raise ValueError("expected posterior samples with shape (chains, draws).")
        thinned = chains[:, ::thinning]
        rhat[name] = _rhat(thinned)
        bulk, tail = _effective_sample_size(thinned)
        ess_bulk[name] = bulk
        ess_tail[name] = tail
        non_converged[name] = bool(rhat[name] > 1.01)
        low_ess[name] = bool(min(bulk, tail) < 400.0)
    return MCMCDiagnostics(
        rhat=rhat,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        non_converged=non_converged,
        low_ess=low_ess,
        # Accepted posterior samples all satisfy the hard wall; proposal-level rejection is internal to NUTS.
        survival_rejection_fraction=0.0,
    )


def _save_trace_plots(samples: dict[str, np.ndarray], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, values in samples.items():
        figure, axis = plt.subplots(figsize=(10, 4))
        draws = np.asarray(values, dtype=np.float64)
        for chain_idx, chain in enumerate(draws):
            axis.plot(chain, linewidth=0.7, alpha=0.8, label=f"chain {chain_idx + 1}")
        axis.set_title(f"Trace Plot: {name}")
        axis.set_xlabel("Draw")
        axis.set_ylabel(name)
        axis.legend(loc="best")
        path = output_dir / f"mcmc_trace_{name}.png"
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)
    return paths


def _observation_sigma_from_series(series: np.ndarray, rel_sigma: float, min_sigma: float) -> float:
    scale = float(np.max(np.abs(np.asarray(series, dtype=np.float64))))
    return max(float(min_sigma), float(rel_sigma) * max(scale, 1.0))


def _numpyro_baseline_model(
    observed_customers: np.ndarray,
    observed_revenue: np.ndarray | None,
    observed_burn: np.ndarray | None,
    observed_cash: np.ndarray | None,
    initial_cash: float,
    dt: float,
    revenue_obs_sigma: float | None,
    burn_obs_sigma: float | None,
    cash_obs_sigma: float | None,
):
    jnp, _, numpyro, dist, _, _, _ = _require_numpyro()

    def model():
        p = numpyro.sample("p", dist.Beta(1.5, 30.0))
        q = numpyro.sample("q", dist.Beta(3.0, 10.0))
        K = numpyro.sample("K", dist.LogNormal(8.5, 0.8))
        gamma = numpyro.sample("gamma", dist.LogNormal(4.2, 0.5))
        # Preserve the intended LogNormal prior on v while enforcing the simulator support v > gamma.
        v_margin_log = numpyro.sample("v_margin_log", dist.Normal(4.0, 1.0))
        v = gamma + jnp.exp(v_margin_log)
        numpyro.factor(
            "v_prior_correction",
            dist.LogNormal(4.6, 0.4).log_prob(v) + v_margin_log - dist.Normal(4.0, 1.0).log_prob(v_margin_log),
        )
        b0 = numpyro.sample("b0", dist.LogNormal(10.8, 0.6))
        sigma_N = numpyro.sample("sigma_N", dist.HalfNormal(50.0))
        customers = jnp.asarray(observed_customers, dtype=jnp.float64)
        current = customers[:-1]
        nxt = customers[1:]
        safe_K = jnp.maximum(K, 1e-12)
        drift = (p + q * current / safe_K) * (safe_K - current)
        residual = nxt - current - drift * dt
        sigma_sq = sigma_N**2
        safe_sigma_sq = jnp.maximum(sigma_sq, 1e-12)

        cash_delta = (v - gamma) * current - b0
        cash_path = jnp.concatenate(
            [
                jnp.asarray([initial_cash], dtype=current.dtype),
                initial_cash + jnp.cumsum(cash_delta * dt),
            ]
        )
        survived = jnp.all(cash_path > 0.0)
        valid = (K > 0.0) & (sigma_N > 0.0) & (jnp.all(current > 0.0))
        safe_current = jnp.maximum(current, 1e-12)
        ll_value = -(customers.size - 1) / 2.0 * jnp.log(2.0 * jnp.pi * dt)
        ll_value -= 0.5 * jnp.sum(jnp.log(safe_sigma_sq * safe_current))
        ll_value -= 0.5 / (safe_sigma_sq * dt) * jnp.sum((residual**2) / safe_current)
        ll_value = jnp.nan_to_num(ll_value, nan=NEGATIVE_HARD_WALL, posinf=NEGATIVE_HARD_WALL, neginf=NEGATIVE_HARD_WALL)
        numpyro.factor("baseline_log_likelihood", jnp.where(valid, ll_value, NEGATIVE_HARD_WALL))
        numpyro.factor("survival_constraint", jnp.where(survived, 0.0, NEGATIVE_HARD_WALL))
        if observed_revenue is not None and revenue_obs_sigma is not None:
            sigma_revenue_obs = jnp.maximum(jnp.asarray(revenue_obs_sigma, dtype=jnp.float64), 1e-12)
            revenue = jnp.asarray(observed_revenue, dtype=jnp.float64)
            revenue_residual = revenue - v * customers
            revenue_ll = -(customers.size / 2.0) * jnp.log(2.0 * jnp.pi * sigma_revenue_obs**2)
            revenue_ll -= 0.5 * jnp.sum((revenue_residual**2) / (sigma_revenue_obs**2))
            numpyro.factor("revenue_observation_likelihood", revenue_ll)
        if observed_burn is not None and burn_obs_sigma is not None:
            sigma_burn_obs = jnp.maximum(jnp.asarray(burn_obs_sigma, dtype=jnp.float64), 1e-12)
            burn = jnp.asarray(observed_burn, dtype=jnp.float64)
            burn_residual = burn - (b0 + gamma * customers)
            burn_ll = -(customers.size / 2.0) * jnp.log(2.0 * jnp.pi * sigma_burn_obs**2)
            burn_ll -= 0.5 * jnp.sum((burn_residual**2) / (sigma_burn_obs**2))
            numpyro.factor("burn_observation_likelihood", burn_ll)
        if observed_cash is not None and cash_obs_sigma is not None:
            sigma_cash_obs = jnp.maximum(jnp.asarray(cash_obs_sigma, dtype=jnp.float64), 1e-12)
            cash = jnp.asarray(observed_cash, dtype=jnp.float64)
            cash_residual = cash - cash_path
            cash_ll = -(cash.size / 2.0) * jnp.log(2.0 * jnp.pi * sigma_cash_obs**2)
            cash_ll -= 0.5 * jnp.sum((cash_residual**2) / (sigma_cash_obs**2))
            numpyro.factor("cash_observation_likelihood", cash_ll)

    return model


def _baseline_init_values(
    observed_customers: np.ndarray,
    initial_cash: float,
    dt: float,
    observed_revenue: np.ndarray | None = None,
    observed_burn: np.ndarray | None = None,
    observed_cash: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    customers = np.asarray(observed_customers, dtype=np.float64)
    current = customers[:-1]
    if np.any(current <= 0.0):
        raise RuntimeError("baseline MCMC requires strictly positive observed customers for all i < T.")
    sigma_guess = max(float(np.std(np.diff(customers))), 1.0)
    K_guess = max(float(np.max(customers) * 1.5), 1_000.0)
    if observed_revenue is not None:
        revenue = np.asarray(observed_revenue, dtype=np.float64)
        revenue_mask = customers > 1e-12
        v_guess = float(np.median(revenue[revenue_mask] / customers[revenue_mask])) if np.any(revenue_mask) else 100.0
    else:
        v_guess = 100.0
    gamma_guess = 40.0
    b0_guess = 50_000.0
    if observed_burn is not None:
        burn = np.asarray(observed_burn, dtype=np.float64)
        design = np.column_stack([customers, np.ones_like(customers)])
        least_squares, _, _, _ = np.linalg.lstsq(design, burn, rcond=None)
        gamma_guess = max(float(least_squares[0]), 1e-6)
        b0_guess = max(float(least_squares[1]), 1e-6)
    elif observed_cash is not None:
        cash = np.asarray(observed_cash, dtype=np.float64)
        cash_velocity = np.diff(cash) / float(dt)
        design = np.column_stack([current, np.ones_like(current)])
        target = v_guess * current - cash_velocity
        least_squares, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        gamma_guess = max(float(least_squares[0]), 1e-6)
        b0_guess = max(float(least_squares[1]), 1e-6)
    delta_guess = max(v_guess - gamma_guess, 1.0)
    theta = {
        "p": 0.03,
        "q": 0.30,
        "K": K_guess,
        "v": gamma_guess + delta_guess,
        "gamma": gamma_guess,
        "b0": b0_guess,
        "sigma_N": sigma_guess,
    }
    while not np.isfinite(baseline_log_likelihood(theta, customers, dt=dt, initial_cash=initial_cash)):
        theta["v"] *= 1.25
        if theta["v"] > 1.0e6:
            raise RuntimeError("failed to construct a valid baseline MCMC initialization.")
    return {
        "p": np.asarray(theta["p"], dtype=np.float64),
        "q": np.asarray(theta["q"], dtype=np.float64),
        "K": np.asarray(theta["K"], dtype=np.float64),
        "gamma": np.asarray(theta["gamma"], dtype=np.float64),
        "v_margin_log": np.asarray(np.log(theta["v"] - theta["gamma"]), dtype=np.float64),
        "b0": np.asarray(theta["b0"], dtype=np.float64),
        "sigma_N": np.asarray(theta["sigma_N"], dtype=np.float64),
    }


def run_baseline_nuts(
    observed: np.ndarray | dict[str, Any],
    *,
    cfg: InferenceConfig,
    seed: int,
    output_dir: str | Path | None = None,
) -> MCMCResult:
    # Baseline NUTS is a small, control-flow-heavy x64 workload. On this machine it is
    # substantially faster on CPU than on the CUDA backend, so default to CPU unless
    # the caller explicitly requests otherwise.
    os.environ["JAX_PLATFORMS"] = cfg.mcmc_platform
    try:
        import numpyro

        numpyro.set_host_device_count(cfg.mcmc_num_chains)
    except Exception:
        pass
    _, random, _, _, MCMC, NUTS, init_to_value = _require_numpyro()

    if isinstance(observed, dict):
        trajectory = np.asarray(observed["trajectory"], dtype=np.float64)
    else:
        trajectory = np.asarray(observed, dtype=np.float64)
    observed_customers = trajectory[:, 0]
    observed_revenue = trajectory[:, 3]
    observed_burn = trajectory[:, 4]
    observed_cash = trajectory[:, 5]
    initial_cash = float(trajectory[0, 5])
    revenue_obs_sigma = _observation_sigma_from_series(
        observed_revenue,
        cfg.baseline_revenue_obs_rel_sigma,
        cfg.baseline_revenue_obs_min_sigma,
    )
    burn_obs_sigma = _observation_sigma_from_series(
        observed_burn,
        cfg.baseline_burn_obs_rel_sigma,
        cfg.baseline_burn_obs_min_sigma,
    )
    cash_obs_sigma = _observation_sigma_from_series(
        observed_cash,
        cfg.baseline_cash_obs_rel_sigma,
        cfg.baseline_cash_obs_min_sigma,
    )
    init_values = _baseline_init_values(
        observed_customers,
        initial_cash,
        cfg.dt,
        observed_revenue=observed_revenue,
        observed_burn=observed_burn,
        observed_cash=observed_cash,
    )

    model = _numpyro_baseline_model(
        observed_customers,
        observed_revenue,
        observed_burn,
        observed_cash,
        initial_cash,
        cfg.dt,
        revenue_obs_sigma,
        burn_obs_sigma,
        cash_obs_sigma,
    )
    chain_method = "parallel" if cfg.mcmc_platform == "cpu" else "vectorized"
    kernel = NUTS(
        model,
        target_accept_prob=cfg.mcmc_target_accept_prob,
        init_strategy=init_to_value(values=init_values),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=cfg.mcmc_num_warmup,
        num_samples=cfg.mcmc_num_samples,
        num_chains=cfg.mcmc_num_chains,
        chain_method=chain_method,
        progress_bar=False,
    )

    start = time.perf_counter()
    mcmc.run(random.PRNGKey(seed))
    elapsed = time.perf_counter() - start

    raw_samples = {name: np.asarray(values, dtype=np.float64) for name, values in mcmc.get_samples(group_by_chain=True).items()}
    posterior_samples = {
        "p": raw_samples["p"],
        "q": raw_samples["q"],
        "K": raw_samples["K"],
        "v": raw_samples["gamma"] + np.exp(raw_samples["v_margin_log"]),
        "gamma": raw_samples["gamma"],
        "b0": raw_samples["b0"],
        "sigma_N": raw_samples["sigma_N"],
    }
    diagnostics = compute_mcmc_diagnostics(posterior_samples, thinning=cfg.mcmc_thinning_for_ess)
    trace_plot_paths = _save_trace_plots(
        posterior_samples,
        Path(output_dir or cfg.outputs_dir / "mcmc_traces"),
    )
    return MCMCResult(
        posterior_samples=posterior_samples,
        diagnostics=diagnostics,
        elapsed_seconds=elapsed,
        trace_plot_paths=trace_plot_paths,
    )


def posterior_array(result: MCMCResult) -> np.ndarray:
    arrays = [result.posterior_samples[name].reshape(-1) for name in BASELINE_PARAM_ORDER]
    return np.column_stack(arrays)


def posterior_predictive_baseline(
    result: MCMCResult,
    observed: dict[str, Any],
    *,
    cfg: InferenceConfig,
    seed: int,
    num_samples: int | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Simulate forecast observables from posterior draws."""

    draws = posterior_array(result)
    valid_mask = np.all(np.isfinite(draws), axis=1) & (draws[:, 3] > draws[:, 4])
    valid_draws = draws[valid_mask]
    if valid_draws.size == 0:
        raise ValueError("no valid posterior draws satisfy v > gamma for baseline predictive simulation.")
    n = valid_draws.shape[0] if num_samples is None else min(int(num_samples), valid_draws.shape[0])
    rng = np.random.default_rng(seed)
    selected = valid_draws[rng.choice(valid_draws.shape[0], size=n, replace=False)]
    last = np.asarray(observed["trajectory"], dtype=np.float64)[-1]
    sims = []
    iterator = tqdm(selected, desc="MCMC posterior predictive", leave=False, disable=not show_progress)
    for idx, theta_values in enumerate(iterator):
        theta = baseline_theta_to_dict(theta_values)
        run = simulate_from_theta(
            theta,
            model="baseline",
            cfg=cfg,
            seed=int(rng.integers(0, 2**31 - 1)) + idx,
            N0=float(last[0]),
            C0=float(last[5]),
            T=cfg.forecast_steps,
        )
        sims.append(observable_matrix(run["trajectory"]))
    predictive = np.stack(sims, axis=0)
    result.posterior_predictive_observables = predictive
    return predictive

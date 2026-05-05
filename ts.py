from __future__ import annotations

import argparse
import csv
import sys
import time
import types
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import core as jax_core

if not hasattr(jax.random, "KeyArray"):
    jax.random.KeyArray = type(jax.random.PRNGKey(0))

if "jax.config" not in sys.modules:
    jax_config_module = types.ModuleType("jax.config")
    jax_config_module.config = jax.config
    sys.modules["jax.config"] = jax_config_module

from composite_tests.bootstrapped_tests import parametric_bootstrap_test
from composite_tests.distributions.toggle_switch import (
    TransformedParams, _small_model, sample_initial_params,
    small_model_default_params, small_toggle_switch,
)
from composite_tests.extra_types import Scalar
from composite_tests.kernels import GaussianKernel, SumKernel
from composite_tests.mmd import MMDStatistic


@dataclass(frozen=True)
class FitConfig:
    iterations: int = 2000
    n_initial_locations: int = 1
    n_optimized_locations: int = 1
    learning_rate: float = 0.04
    ridge: float = 1e-3
    ell_initial: float = 1000.0
    ell_final: float = 80.0
    ell_decay: float = 0.96
    simulator_n: int | None = None
    verbose: bool = False


def params_to_free_vector(p: TransformedParams) -> jax.Array:
    return jnp.array([p.alpha_1, p.alpha_2, p.beta_1, p.beta_2, p.mu, p.sigma, p.gamma])


def free_vector_to_params(theta: jax.Array) -> TransformedParams:
    fixed = small_model_default_params
    return TransformedParams(
        alpha_1=theta[0], alpha_2=theta[1], beta_1=theta[2], beta_2=theta[3],
        mu=theta[4], sigma=theta[5], gamma=theta[6],
        kappa_1=fixed.kappa_1, kappa_2=fixed.kappa_2,
        delta_1=fixed.delta_1, delta_2=fixed.delta_2,
    )


def original_toggle_kernel() -> SumKernel:
    return SumKernel([
        GaussianKernel(20.0), GaussianKernel(40.0), GaussianKernel(80.0),
        GaussianKernel(100.0), GaussianKernel(130.0), GaussianKernel(200.0),
        GaussianKernel(400.0), GaussianKernel(800.0), GaussianKernel(1000.0),
    ])


@jax.jit
def gaussian_kernel_matrix(x, y, ell):
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    sqdist = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-0.5 * sqdist / (ell**2))


@jax.jit
def mmd_v_stat_gaussian(xs, ys, ell):
    return (
        gaussian_kernel_matrix(xs, xs, ell).mean()
        - 2.0 * gaussian_kernel_matrix(xs, ys, ell).mean()
        + gaussian_kernel_matrix(ys, ys, ell).mean()
    )


@jax.jit
def adaptive_lengthscale(t, ell_initial, ell_final, ell_decay):
    return jnp.maximum(ell_final, ell_initial * (ell_decay**t))


@partial(jax.jit, static_argnames=("T", "n"))
def simulate_toggle_from_theta_fast(rng, theta, *, T, n):
    params = free_vector_to_params(theta)
    rngs = jax.random.split(rng, n)
    samples = jax.vmap(_small_model, in_axes=(0, None, None))(rngs, params, T)
    return samples.reshape((n, 1))


@partial(jax.jit, static_argnames=("n_initial_locations",))
def initial_theta_batch_fast(rng, *, n_initial_locations):
    rngs = jax.random.split(rng, n_initial_locations)
    params = jax.vmap(sample_initial_params)(rngs)
    return jax.vmap(params_to_free_vector)(params)


# -----------------------------------------------------------------------------
# PGD
# -----------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("T", "n_sim"))
def pgd_step(rng_objective, theta, ys, t, *, T, n_sim,
             learning_rate, ridge, ell_initial, ell_final, ell_decay):
    ell_t = adaptive_lengthscale(t, ell_initial, ell_final, ell_decay)

    def samples_for_theta(th):
        return simulate_toggle_from_theta_fast(rng_objective, th, T=T, n=n_sim)

    xs = samples_for_theta(theta)
    jac = jax.jacrev(samples_for_theta)(theta)
    jac = jac.reshape((n_sim, -1, theta.shape[0]))

    jac_norms = jnp.linalg.norm(jac, axis=(1, 2), keepdims=True) + 1e-8
    jac_normalized = jac / jac_norms

    _, grad_xs = jax.value_and_grad(mmd_v_stat_gaussian)(xs, ys, ell_t)
    grad_xs = grad_xs.reshape((n_sim, -1))
    grad_theta = jnp.einsum("ndp,nd->p", jac_normalized, grad_xs)

    hessian_like = jnp.einsum("ndp,ndq->pq", jac_normalized, jac_normalized) / n_sim
    preconditioner = hessian_like + ridge * jnp.eye(theta.shape[0])

    direction = jnp.linalg.solve(preconditioner, grad_theta)
    theta_next = theta - learning_rate * direction

    xs_next = samples_for_theta(theta_next)
    loss_val = mmd_v_stat_gaussian(xs_next, ys, ell_t)

    return theta_next, loss_val


@partial(jax.jit, static_argnames=("T", "n_sim", "iterations"))
def fit_one_initialization_pgd(rng, theta0, ys, *, T, n_sim, iterations,
                                learning_rate, ridge, ell_initial, ell_final, ell_decay):
    _, rng_objective = jax.random.split(rng)

    def body(carry, t):
        theta, rng = carry
        rng, rng_step = jax.random.split(rng)
        theta_next, loss = pgd_step(
            rng_step, theta, ys, t,
            T=T, n_sim=n_sim, learning_rate=learning_rate, ridge=ridge,
            ell_initial=ell_initial, ell_final=ell_final, ell_decay=ell_decay,
        )
        return (theta_next, rng), loss

    (theta_final, _), losses = jax.lax.scan(
        body, (theta0, rng_objective), jnp.arange(iterations)
    )
    return theta_final, losses[-1]


@partial(jax.jit, static_argnames=("T", "n_sim", "iterations",
                                    "n_initial_locations", "n_optimized_locations"))
def estimate_theta_pgd(rng, ys, *, T, n_sim, iterations, n_initial_locations,
                       n_optimized_locations, learning_rate, ridge,
                       ell_initial, ell_final, ell_decay):
    rng, rng_init = jax.random.split(rng)
    init_thetas = initial_theta_batch_fast(rng_init, n_initial_locations=n_initial_locations)

    def screen_one(local_rng, theta):
        xs = simulate_toggle_from_theta_fast(local_rng, theta, T=T, n=n_sim)
        return mmd_v_stat_gaussian(xs, ys, ell_final)

    rng, rng_screen = jax.random.split(rng)
    screen_rngs = jax.random.split(rng_screen, n_initial_locations)
    screen_losses = jax.vmap(screen_one)(screen_rngs, init_thetas)
    best_idx = jnp.argsort(screen_losses)[:n_optimized_locations]
    best_inits = init_thetas[best_idx]

    rng, rng_opt = jax.random.split(rng)
    opt_rngs = jax.random.split(rng_opt, n_optimized_locations)

    def optimize_one(local_rng, theta0):
        return fit_one_initialization_pgd(
            local_rng, theta0, ys, T=T, n_sim=n_sim, iterations=iterations,
            learning_rate=learning_rate, ridge=ridge, ell_initial=ell_initial,
            ell_final=ell_final, ell_decay=ell_decay,
        )

    final_thetas, final_losses = jax.vmap(optimize_one)(opt_rngs, best_inits)
    best = jnp.argmin(final_losses)
    return final_thetas[best], final_losses[best]


# -----------------------------------------------------------------------------
# SGD
# -----------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("T", "n_sim"))
def sgd_step(rng, theta, ys, *, T, n_sim, learning_rate, ell_final):
    def loss_fn(th):
        return mmd_v_stat_gaussian(
            simulate_toggle_from_theta_fast(rng, th, T=T, n=n_sim), ys, ell_final
        )
    loss_val, grad = jax.value_and_grad(loss_fn)(theta)
    grad = jnp.clip(grad, -1.0, 1.0)  # prevent divergence
    return theta - learning_rate * grad, loss_val


@partial(jax.jit, static_argnames=("T", "n_sim", "iterations"))
def fit_one_initialization_sgd(rng, theta0, ys, *, T, n_sim, iterations,
                                learning_rate, ell_final):
    def body(carry, _):
        theta, rng = carry
        rng, rng_step = jax.random.split(rng)
        theta_next, loss = sgd_step(
            rng_step, theta, ys, T=T, n_sim=n_sim,
            learning_rate=learning_rate, ell_final=ell_final,
        )
        return (theta_next, rng), loss

    (theta_final, _), losses = jax.lax.scan(
        body, (theta0, rng), jnp.arange(iterations)
    )
    return theta_final, losses[-1]


@partial(jax.jit, static_argnames=("T", "n_sim", "iterations",
                                    "n_initial_locations", "n_optimized_locations"))
def estimate_theta_sgd(rng, ys, *, T, n_sim, iterations, n_initial_locations,
                       n_optimized_locations, learning_rate, ell_final):
    rng, rng_init = jax.random.split(rng)
    init_thetas = initial_theta_batch_fast(rng_init, n_initial_locations=n_initial_locations)

    def screen_one(local_rng, theta):
        xs = simulate_toggle_from_theta_fast(local_rng, theta, T=T, n=n_sim)
        return mmd_v_stat_gaussian(xs, ys, ell_final)

    rng, rng_screen = jax.random.split(rng)
    screen_rngs = jax.random.split(rng_screen, n_initial_locations)
    screen_losses = jax.vmap(screen_one)(screen_rngs, init_thetas)
    best_idx = jnp.argsort(screen_losses)[:n_optimized_locations]
    best_inits = init_thetas[best_idx]

    rng, rng_opt = jax.random.split(rng)
    opt_rngs = jax.random.split(rng_opt, n_optimized_locations)

    def optimize_one(local_rng, theta0):
        return fit_one_initialization_sgd(
            local_rng, theta0, ys, T=T, n_sim=n_sim, iterations=iterations,
            learning_rate=learning_rate, ell_final=ell_final,
        )

    final_thetas, final_losses = jax.vmap(optimize_one)(opt_rngs, best_inits)
    best = jnp.argmin(final_losses)
    return final_thetas[best], final_losses[best]


# -----------------------------------------------------------------------------
# Estimator classes
# -----------------------------------------------------------------------------

class FastPGDEstimator:
    def __init__(self, T, config):
        self.T = T
        self.config = config
        self.kernel = GaussianKernel(config.ell_final)
        # self.kernel = original_toggle_kernel()
        self.last_fit_time = self.last_final_mmd = self.last_theta = None
        self.fit_calls = 0

    def __call__(self, rng, ys):
        self.fit_calls += 1
        start = time.perf_counter()
        ys = jnp.atleast_2d(ys)
        n_sim = self.config.simulator_n or ys.shape[0]

        theta_vec, final_mmd = estimate_theta_pgd(
            rng, ys, T=self.T, n_sim=int(n_sim),
            iterations=self.config.iterations,
            n_initial_locations=self.config.n_initial_locations,
            n_optimized_locations=self.config.n_optimized_locations,
            learning_rate=self.config.learning_rate,
            ridge=self.config.ridge,
            ell_initial=self.config.ell_initial,
            ell_final=self.config.ell_final,
            ell_decay=self.config.ell_decay,
        )
        theta_hat = free_vector_to_params(theta_vec)

        if not isinstance(final_mmd, jax_core.Tracer):
            final_mmd.block_until_ready()
            if self.fit_calls == 1:
                self.last_fit_time = time.perf_counter() - start
                self.last_final_mmd = float(final_mmd)
                self.last_theta = theta_hat
            print(f"pgd fit #{self.fit_calls}: mmd={float(final_mmd):.6f}", flush=True)

        return theta_hat

    @property
    def name(self): return "fast_pgd"


class FastSGDEstimator:
    def __init__(self, T, config):
        self.T = T
        self.config = config
        # self.kernel = original_toggle_kernel()
        self.kernel = GaussianKernel(config.ell_final)
        self.last_fit_time = self.last_final_mmd = self.last_theta = None
        self.fit_calls = 0

    def __call__(self, rng, ys):
        self.fit_calls += 1
        start = time.perf_counter()
        ys = jnp.atleast_2d(ys)
        n_sim = self.config.simulator_n or ys.shape[0]

        theta_vec, final_mmd = estimate_theta_sgd(
            rng, ys, T=self.T, n_sim=int(n_sim),
            iterations=self.config.iterations,
            n_initial_locations=self.config.n_initial_locations,
            n_optimized_locations=self.config.n_optimized_locations,
            learning_rate=self.config.learning_rate,
            ell_final=self.config.ell_final,
        )
        theta_hat = free_vector_to_params(theta_vec)

        if not isinstance(final_mmd, jax_core.Tracer):
            final_mmd.block_until_ready()
            if self.fit_calls == 1:
                self.last_fit_time = time.perf_counter() - start
                self.last_final_mmd = float(final_mmd)
                self.last_theta = theta_hat
            print(f"sgd fit #{self.fit_calls}: mmd={float(final_mmd):.6f}", flush=True)

        return theta_hat

    @property
    def name(self): return "fast_sgd"


# -----------------------------------------------------------------------------
# GoF runner
# -----------------------------------------------------------------------------

def safe_float(x):
    try: return float(jax.device_get(x))
    except: return None


def run_one_gof(rng, *, T, n, config, n_bootstrap_samples, alpha, T_true=None, method="pgd"):
    null_family = small_toggle_switch(T)

    if method == "pgd":
        estimator = FastPGDEstimator(T, config)
    elif method == "sgd":
        estimator = FastSGDEstimator(T, config)
    else:
        raise ValueError(f"Unknown method: {method}")

    statistic = MMDStatistic(estimator.kernel, null_family)
    if T_true is None: T_true = T
    true_family = small_toggle_switch(T_true)

    rng, rng_data = jax.random.split(rng)
    ys = true_family.sample_with_params(rng_data, small_model_default_params, n)

    if config.verbose:
        rng, rng_diag = jax.random.split(rng)
        print(f"Diagnostic fit ({method})...", flush=True)
        estimator(rng_diag, ys)
        print({"diagnostic_fit_time": estimator.last_fit_time,
               "diagnostic_final_mmd": estimator.last_final_mmd}, flush=True)
        estimator.fit_calls = 0
        estimator.last_fit_time = None
        estimator.last_final_mmd = None
        estimator.last_theta = None

    rng, rng_test = jax.random.split(rng)
    start = time.perf_counter()
    result = parametric_bootstrap_test(
        rng_test, ys, estimator, null_family, statistic,
        n_bootstrap_samples=n_bootstrap_samples,
        save_null_distribution=True,
        level=alpha,
    )
    total_time = time.perf_counter() - start
    pvalue = float(jnp.mean(result.bootstrapped_test_stats >= result.test_statistic))
    threshold = safe_float(getattr(result, 'threshold', float('nan')))

    return {
        "T": T, "T_true": T_true, "n": n, "method": method,
        "estimator": estimator.name,
        "pvalue": pvalue, "reject": bool(result.reject_null), "alpha": alpha,
        "test_statistic": float(result.test_statistic),
        "threshold": threshold,
        "total_time_sec": total_time, "fit_time_sec": estimator.last_fit_time,
        "final_fit_mmd": safe_float(estimator.last_final_mmd),
    }


def write_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def run_grid(args):
    config = FitConfig(
        iterations=args.iterations, n_initial_locations=args.n_inits,
        n_optimized_locations=args.n_opt, learning_rate=args.learning_rate,
        ridge=args.ridge, ell_initial=args.ell_initial, ell_final=args.ell_final,
        ell_decay=args.ell_decay, simulator_n=args.simulator_n, verbose=args.verbose,
    )
    all_rows = []
    root_rng = jax.random.PRNGKey(args.rng_seed)
    for T in args.Ts:
        for seed in range(args.seeds):
            for method in args.methods:
                root_rng, run_rng = jax.random.split(root_rng)
                print(f"Running {method.upper()} T={T}, seed={seed}", flush=True)
                row = run_one_gof(run_rng, T=T, n=args.n, config=config,
                                  n_bootstrap_samples=args.bootstrap, alpha=args.alpha,
                                  T_true=args.T_true, method=method)
                row["seed"] = seed
                all_rows.append(row)
                print(row, flush=True)
                write_rows(args.output, all_rows)
    return all_rows


def summarize_rows(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["T"]), []).append(row)
    print("\nSummary\n-------")
    for (method, T), rs in sorted(groups.items()):
        print({"method": method, "T": T,
               "reject_rate": sum(r["reject"] for r in rs) / len(rs),
               "mean_pvalue": sum(r["pvalue"] for r in rs) / len(rs),
               "mean_total_time_sec": sum(r["total_time_sec"] for r in rs) / len(rs)})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ts", type=int, nargs="+", default=[10])
    parser.add_argument("--T-true", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--methods", nargs="+", choices=["pgd", "sgd"], default=["pgd"])
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--rng-seed", type=int, default=1489048490)
    parser.add_argument("--output", type=str, default="results/sgd.csv")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--n-inits", type=int, default=1)
    parser.add_argument("--n-opt", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.2) #0.2
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--ell-initial", type=float, default=1000.0)
    parser.add_argument("--ell-final", type=float, default=80.0)
    parser.add_argument("--ell-decay", type=float, default=0.96)
    parser.add_argument("--simulator-n", type=int, default=None) 
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = run_grid(args)
    summarize_rows(rows)
    print(f"\nSaved results to {args.output}")
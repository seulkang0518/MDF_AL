import os
os.environ["JAX_PLATFORMS"] = "cpu"
from functools import partial
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.nn as jnn
import jax.numpy as jnp


# ============================================================
# 1. Stochastic Lotka-Volterra simulator
# ============================================================
# We follow the paper's experiment structure: the dynamic parameters theta1
# are fixed and the unknown parameter theta2 is the initial condition.
DEFAULT_THETA1 = np.array([5.0, 0.025, 6.0], dtype=np.float64)
DEFAULT_THETA_TRUE = np.array([100.0, 120.0], dtype=np.float64)
DEFAULT_THETA_BAD = np.array([50.0, 50.0], dtype=np.float64)


def theta_to_phi(theta):
    theta = jnp.asarray(theta, dtype=jnp.float64)
    return jnp.log(jnp.expm1(jnp.maximum(theta - 1e-6, 1e-12)))


def phi_to_theta(phi):
    return jnn.softplus(phi) + 1e-6


def lv_drift_and_rates(x, theta1):
    x = jnp.maximum(x, 1e-9)
    theta11, theta12, theta13 = theta1
    x1, x2 = x
    rates = jnp.array(
        [
            theta11 * x1,
            theta12 * x1 * x2,
            theta13 * x2,
        ],
        dtype=jnp.float64,
    )
    drift = jnp.array(
        [
            rates[0] - rates[1],
            rates[1] - rates[2],
        ],
        dtype=jnp.float64,
    )
    return drift, rates


def lv_trajectory_flat(theta2, noises, T=1.0, theta1=DEFAULT_THETA1):
    theta1 = jnp.asarray(theta1, dtype=jnp.float64)
    theta2 = jnp.asarray(theta2, dtype=jnp.float64)
    dt = jnp.asarray(T / noises.shape[0], dtype=jnp.float64)
    sqrt_dt = jnp.sqrt(dt)
    stoich = jnp.array(
        [
            [1.0, 0.0],
            [-1.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=jnp.float64,
    )

    def one_step(x, z):
        drift, rates = lv_drift_and_rates(x, theta1)
        reaction_noise = jnp.sqrt(jnp.maximum(rates, 0.0)) * sqrt_dt * z
        stochastic = reaction_noise @ stoich
        x_next = jnp.maximum(x + drift * dt + stochastic, 1e-9)
        return x_next, x_next

    _, states = jax.lax.scan(one_step, theta2, noises)
    # One simulated observation is the terminal two-species state at T=1.
    return states[-1]


def simulate_lv_samples_from_noises(theta2, noises, T=1.0, theta1=DEFAULT_THETA1):
    return jax.vmap(lv_trajectory_flat, in_axes=(None, 0, None, None))(
        theta2,
        noises,
        T,
        jnp.asarray(theta1, dtype=jnp.float64),
    )


def sample_lv(key, theta2, n, num_steps=100, T=1.0, theta1=DEFAULT_THETA1):
    noises = jax.random.normal(key, shape=(n, num_steps, 3), dtype=jnp.float64)
    samples = simulate_lv_samples_from_noises(theta2, noises, T=T, theta1=theta1)
    return noises, samples


def make_observed_data(
    seed,
    theta_true=DEFAULT_THETA_TRUE,
    theta_bad=DEFAULT_THETA_BAD,
    m_obs=100,
    corruption=0.0,
    num_steps=100,
    T=1.0,
    theta1=DEFAULT_THETA1,
):
    key = jax.random.PRNGKey(seed)
    key_true, key_bad = jax.random.split(key)
    n_bad = int(round(float(corruption) * int(m_obs)))
    n_bad = max(0, min(int(m_obs), n_bad))
    n_true = int(m_obs) - n_bad

    parts = []
    if n_true > 0:
        _, y_true = sample_lv(
            key_true,
            jnp.asarray(theta_true, dtype=jnp.float64),
            n_true,
            num_steps=num_steps,
            T=T,
            theta1=theta1,
        )
        parts.append(y_true)
    if n_bad > 0:
        _, y_bad = sample_lv(
            key_bad,
            jnp.asarray(theta_bad, dtype=jnp.float64),
            n_bad,
            num_steps=num_steps,
            T=T,
            theta1=theta1,
        )
        parts.append(y_bad)

    return jnp.concatenate(parts, axis=0)


# ============================================================
# 2. MMD objective and gradients wrt simulated trajectories
# ============================================================
def sq_dists(x, y):
    x2 = jnp.sum(x**2, axis=1, keepdims=True)
    y2 = jnp.sum(y**2, axis=1, keepdims=True).T
    return jnp.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)


def gaussian_kernel(x, y, ell):
    return jnp.exp(-0.5 * sq_dists(x, y) / (ell**2))


def mmd2_vstat(x, y, ell):
    k_xx = gaussian_kernel(x, x, ell)
    k_yy = gaussian_kernel(y, y, ell)
    k_xy = gaussian_kernel(x, y, ell)
    return jnp.mean(k_xx) + jnp.mean(k_yy) - 2.0 * jnp.mean(k_xy)


def witness_gradient_empirical(x, y, ell):
    diff_xx = x[:, None, :] - x[None, :, :]
    k_xx = jnp.exp(-0.5 * jnp.sum(diff_xx**2, axis=2) / (ell**2))
    grad_emp = jnp.mean(k_xx[:, :, None] * (-diff_xx) / (ell**2), axis=1)

    diff_yx = y[:, None, :] - x[None, :, :]
    k_yx = jnp.exp(-0.5 * jnp.sum(diff_yx**2, axis=2) / (ell**2))
    grad_tar = jnp.mean(k_yx[:, :, None] * diff_yx / (ell**2), axis=0)
    return grad_emp - grad_tar


def make_adaptive_ell_schedule(n_steps, ell0, ell_min, decay):
    ts = np.arange(n_steps, dtype=np.float64)
    return np.maximum(ell_min, ell0 * (decay**ts))


# ============================================================
# 3. One-step optimizers
# ============================================================
@partial(jax.jit, static_argnums=(2, 3))
def eval_loss_full(
    theta2,
    key_eval,
    n_eval_model,
    num_steps,
    T,
    theta1,
    y_obs_full,
    scale_mean,
    scale_std,
    ell_t,
):
    _, x_eval = sample_lv(
        key_eval,
        theta2,
        n_eval_model,
        num_steps=num_steps,
        T=T,
        theta1=theta1,
    )
    x_eval = (x_eval - scale_mean) / scale_std
    y_obs_full = (y_obs_full - scale_mean) / scale_std
    return mmd2_vstat(x_eval, y_obs_full, ell_t)


@partial(jax.jit, static_argnums=(3, 4))
def sgd_step_phi(
    phi,
    key_model,
    y_batch,
    n_model,
    num_steps,
    T,
    theta1,
    scale_mean,
    scale_std,
    ell_t,
    gamma_t,
):
    noises = jax.random.normal(key_model, shape=(n_model, num_steps, 3), dtype=jnp.float64)
    y_batch = (y_batch - scale_mean) / scale_std

    def objective(ph):
        theta2 = phi_to_theta(ph)
        x_raw = simulate_lv_samples_from_noises(theta2, noises, T=T, theta1=theta1)
        x_model = (x_raw - scale_mean) / scale_std
        return mmd2_vstat(x_model, y_batch, ell_t)

    train_loss, grad_phi = jax.value_and_grad(objective)(phi)
    theta2 = phi_to_theta(phi)
    phi_new = phi - gamma_t * grad_phi
    return phi_new, train_loss, grad_phi, phi_to_theta(phi_new) - theta2


@partial(jax.jit, static_argnums=(3, 4))
def pgd_step_phi(
    phi,
    key_model,
    y_batch,
    n_model,
    num_steps,
    T,
    theta1,
    scale_mean,
    scale_std,
    ell_t,
    gamma_t,
    lambda_t,
):
    noises = jax.random.normal(key_model, shape=(n_model, num_steps, 3), dtype=jnp.float64)
    theta2 = phi_to_theta(phi)
    x_raw = simulate_lv_samples_from_noises(theta2, noises, T=T, theta1=theta1)
    x_model = (x_raw - scale_mean) / scale_std
    y_batch = (y_batch - scale_mean) / scale_std
    grad_f = witness_gradient_empirical(x_model, y_batch, ell_t)

    jac_fun = jax.jacrev(lambda ph, noise: lv_trajectory_flat(phi_to_theta(ph), noise, T, theta1))
    J_phi = jax.vmap(jac_fun, in_axes=(None, 0))(phi, noises) / scale_std[None, :, None]
    grad_phi = jnp.einsum("ndp,nd->p", J_phi, grad_f) / n_model
    A = jnp.einsum("ndp,ndq->pq", J_phi, J_phi) / n_model
    A = A + lambda_t * jnp.eye(phi.shape[0], dtype=phi.dtype)
    direction = jnp.linalg.solve(A, grad_phi)

    phi_new = phi - gamma_t * direction
    train_loss = mmd2_vstat(x_model, y_batch, ell_t)
    return phi_new, train_loss, direction, phi_to_theta(phi_new) - theta2


# ============================================================
# 4. Method runners
# ============================================================
def _sample_target_batch(key, y_obs_full, target_batch_size):
    if target_batch_size is None or target_batch_size >= y_obs_full.shape[0]:
        return y_obs_full
    idx = jax.random.randint(
        key,
        shape=(target_batch_size,),
        minval=0,
        maxval=y_obs_full.shape[0],
    )
    return y_obs_full[idx]


def run_sgd(
    seed,
    theta0,
    y_obs_full,
    scale_mean,
    scale_std,
    n_model=50,
    target_batch_size=100,
    n_steps=500,
    gamma=1e-2,
    ell_fixed=30.0,
    ell_eval=30.0,
    num_steps=100,
    T=1.0,
    theta1=DEFAULT_THETA1,
    n_eval_model=200,
    history_every=1,
    print_every=100,
):
    key = jax.random.PRNGKey(seed + 1000)
    phi = theta_to_phi(theta0)
    theta1 = jnp.asarray(theta1, dtype=jnp.float64)
    theta_history = []
    eval_loss_history = []
    train_loss_history = []
    history_steps = []
    direction_history = []
    theta_delta_history = []
    last_train_loss = None
    last_eval_loss = None

    for t in range(n_steps):
        key, key_batch, key_model, key_eval = jax.random.split(key, 4)
        y_batch = _sample_target_batch(key_batch, y_obs_full, target_batch_size)
        phi, train_loss, grad_phi, theta_delta = sgd_step_phi(
            phi,
            key_model,
            y_batch,
            n_model,
            num_steps,
            jnp.asarray(T, dtype=jnp.float64),
            theta1,
            scale_mean,
            scale_std,
            jnp.asarray(ell_fixed, dtype=jnp.float64),
            jnp.asarray(gamma, dtype=jnp.float64),
        )
        last_train_loss = train_loss

        if (t % history_every == 0) or (t == n_steps - 1):
            theta = phi_to_theta(phi)
            eval_loss = eval_loss_full(
                theta,
                key_eval,
                n_eval_model,
                num_steps,
                jnp.asarray(T, dtype=jnp.float64),
                theta1,
                y_obs_full,
                scale_mean,
                scale_std,
                jnp.asarray(ell_eval, dtype=jnp.float64),
            )
            last_eval_loss = eval_loss
            history_steps.append(t)
            train_loss_history.append(float(train_loss))
            eval_loss_history.append(float(eval_loss))
            theta_history.append(np.array(theta, dtype=np.float64))
            direction_history.append(np.array(grad_phi, dtype=np.float64))
            theta_delta_history.append(np.array(theta_delta, dtype=np.float64))

        if print_every and ((t % print_every == 0) or (t == n_steps - 1)):
            print(
                f"[LV SGD] step={t:4d} theta={np.array(phi_to_theta(phi))} "
                f"train={float(train_loss):.6e} eval={float(last_eval_loss):.6e}"
            )

    return {
        "theta_final": np.array(phi_to_theta(phi), dtype=np.float64),
        "train_loss_final": float(last_train_loss),
        "eval_loss_final": float(last_eval_loss),
        "history_steps": np.asarray(history_steps, dtype=np.int32),
        "train_loss_history": np.asarray(train_loss_history, dtype=np.float64),
        "eval_loss_history": np.asarray(eval_loss_history, dtype=np.float64),
        "theta_history": np.asarray(theta_history, dtype=np.float64),
        "direction_history": np.asarray(direction_history, dtype=np.float64),
        "theta_delta_history": np.asarray(theta_delta_history, dtype=np.float64),
    }


def run_pgd(
    seed,
    theta0,
    y_obs_full,
    scale_mean,
    scale_std,
    n_model=50,
    n_steps=500,
    gamma=1e-2,
    lambda_scale=1e-3,
    ell0=100.0,
    ell_min=30.0,
    decay=0.995,
    ell_eval=30.0,
    num_steps=100,
    T=1.0,
    theta1=DEFAULT_THETA1,
    n_eval_model=200,
    history_every=1,
    print_every=100,
):
    key = jax.random.PRNGKey(seed + 2000)
    phi = theta_to_phi(theta0)
    theta1 = jnp.asarray(theta1, dtype=jnp.float64)
    ell_schedule = make_adaptive_ell_schedule(n_steps, ell0, ell_min, decay)
    theta_history = []
    eval_loss_history = []
    train_loss_history = []
    history_steps = []
    direction_history = []
    theta_delta_history = []
    last_train_loss = None
    last_eval_loss = None

    for t in range(n_steps):
        key, key_model, key_eval = jax.random.split(key, 3)
        ell_t = float(ell_schedule[t])
        gamma_t = jnp.asarray(gamma, dtype=jnp.float64)
        lambda_t = jnp.asarray(
            lambda_scale * ((float(gamma_t) * (float(ell_min) ** -1.0) * (ell_t**-2.0)) ** (2.0 / 5.0)),
            dtype=jnp.float64,
        )
        phi, train_loss, direction, theta_delta = pgd_step_phi(
            phi,
            key_model,
            y_obs_full,
            n_model,
            num_steps,
            jnp.asarray(T, dtype=jnp.float64),
            theta1,
            scale_mean,
            scale_std,
            jnp.asarray(ell_t, dtype=jnp.float64),
            gamma_t,
            lambda_t,
        )
        last_train_loss = train_loss

        if (t % history_every == 0) or (t == n_steps - 1):
            theta = phi_to_theta(phi)
            eval_loss = eval_loss_full(
                theta,
                key_eval,
                n_eval_model,
                num_steps,
                jnp.asarray(T, dtype=jnp.float64),
                theta1,
                y_obs_full,
                scale_mean,
                scale_std,
                jnp.asarray(ell_eval, dtype=jnp.float64),
            )
            last_eval_loss = eval_loss
            history_steps.append(t)
            train_loss_history.append(float(train_loss))
            eval_loss_history.append(float(eval_loss))
            theta_history.append(np.array(theta, dtype=np.float64))
            direction_history.append(np.array(direction, dtype=np.float64))
            theta_delta_history.append(np.array(theta_delta, dtype=np.float64))

        if print_every and ((t % print_every == 0) or (t == n_steps - 1)):
            print(
                f"[LV PGD] step={t:4d} ell={ell_t:.4f} theta={np.array(phi_to_theta(phi))} "
                f"train={float(train_loss):.6e} eval={float(last_eval_loss):.6e}"
            )

    return {
        "theta_final": np.array(phi_to_theta(phi), dtype=np.float64),
        "train_loss_final": float(last_train_loss),
        "eval_loss_final": float(last_eval_loss),
        "history_steps": np.asarray(history_steps, dtype=np.int32),
        "train_loss_history": np.asarray(train_loss_history, dtype=np.float64),
        "eval_loss_history": np.asarray(eval_loss_history, dtype=np.float64),
        "theta_history": np.asarray(theta_history, dtype=np.float64),
        "direction_history": np.asarray(direction_history, dtype=np.float64),
        "theta_delta_history": np.asarray(theta_delta_history, dtype=np.float64),
    }


# ============================================================
# 5. Combined experiment and aggregation
# ============================================================
def run_one_seed(
    seed=0,
    theta_true=DEFAULT_THETA_TRUE,
    theta_bad=DEFAULT_THETA_BAD,
    theta0=np.array([90.0, 90.0], dtype=np.float64),
    corruption=0.1,
    m_obs=100,
    n_model=50,
    n_steps=500,
    sgd_n_steps=None,
    pgd_n_steps=None,
    target_batch_size=100,
    num_steps=100,
    T=1.0,
    theta1=DEFAULT_THETA1,
    ell_fixed=30.0,
    ell_eval=30.0,
    pgd_gamma=1e-2,
    pgd_lambda_scale=1e-3,
    pgd_ell0=100.0,
    pgd_ell_min=30.0,
    pgd_decay=0.995,
    n_eval_model=200,
    history_every=1,
    print_every=100,
    run_plain_sgd=True,
    sgd_gamma=1e-2,
    standardize=False,
    scale_eps=1e-6,
):
    theta_true = np.asarray(theta_true, dtype=np.float64)
    theta_bad = np.asarray(theta_bad, dtype=np.float64)
    theta0 = np.asarray(theta0, dtype=np.float64)
    theta1 = np.asarray(theta1, dtype=np.float64)
    sgd_n_steps = int(n_steps if sgd_n_steps is None else sgd_n_steps)
    pgd_n_steps = int(n_steps if pgd_n_steps is None else pgd_n_steps)

    y_obs_full = make_observed_data(
        seed=seed,
        theta_true=theta_true,
        theta_bad=theta_bad,
        m_obs=m_obs,
        corruption=corruption,
        num_steps=num_steps,
        T=T,
        theta1=theta1,
    )
    if standardize:
        scale_mean = jnp.mean(y_obs_full, axis=0)
        scale_std = jnp.std(y_obs_full, axis=0) + jnp.asarray(scale_eps, dtype=jnp.float64)
    else:
        scale_mean = jnp.zeros((y_obs_full.shape[1],), dtype=jnp.float64)
        scale_std = jnp.ones((y_obs_full.shape[1],), dtype=jnp.float64)

    result = {
        "theta_true": theta_true,
        "theta_bad": theta_bad,
        "theta0": theta0,
        "corruption": np.asarray(corruption, dtype=np.float64),
        "standardize": np.asarray(standardize, dtype=np.bool_),
        "scale_mean": np.asarray(scale_mean, dtype=np.float64),
        "scale_std": np.asarray(scale_std, dtype=np.float64),
    }

    if run_plain_sgd:
        sgd_res = run_sgd(
            seed=seed,
            theta0=theta0,
            y_obs_full=y_obs_full,
            scale_mean=scale_mean,
            scale_std=scale_std,
            n_model=n_model,
            target_batch_size=target_batch_size,
            n_steps=sgd_n_steps,
            gamma=sgd_gamma,
            ell_fixed=ell_fixed,
            ell_eval=ell_eval,
            num_steps=num_steps,
            T=T,
            theta1=theta1,
            n_eval_model=n_eval_model,
            history_every=history_every,
            print_every=print_every,
        )
        result.update(_prefix_result("sgd", sgd_res))

    pgd_res = run_pgd(
        seed=seed,
        theta0=theta0,
        y_obs_full=y_obs_full,
        scale_mean=scale_mean,
        scale_std=scale_std,
        n_model=n_model,
        n_steps=pgd_n_steps,
        gamma=pgd_gamma,
        lambda_scale=pgd_lambda_scale,
        ell0=pgd_ell0,
        ell_min=pgd_ell_min,
        decay=pgd_decay,
        ell_eval=ell_eval,
        num_steps=num_steps,
        T=T,
        theta1=theta1,
        n_eval_model=n_eval_model,
        history_every=history_every,
        print_every=print_every,
    )
    result.update(_prefix_result("pgd", pgd_res))
    return result


def _prefix_result(prefix, result):
    return {f"{prefix}_{key}": value for key, value in result.items()}


def _stack(values, dtype=np.float64):
    return np.asarray(values, dtype=dtype)


def run_experiment(
    seeds=range(5),
    output_path=None,
    **kwargs,
):
    per_seed = []
    for seed in seeds:
        print(f"\n=== Lotka-Volterra seed {seed} ===")
        per_seed.append(run_one_seed(seed=seed, **kwargs))

    method_names = ["pgd"]
    if any("sgd_theta_final" in res for res in per_seed):
        method_names.insert(0, "sgd")

    results = {
        "seeds": np.asarray(list(seeds), dtype=np.int32),
        "theta_true": per_seed[0]["theta_true"],
        "theta_bad": per_seed[0]["theta_bad"],
        "theta0": per_seed[0]["theta0"],
        "corruption": per_seed[0]["corruption"],
        "standardize": per_seed[0]["standardize"],
        "scale_mean": per_seed[0]["scale_mean"],
        "scale_std": per_seed[0]["scale_std"],
    }

    for method in method_names:
        theta_finals = _stack([res[f"{method}_theta_final"] for res in per_seed])
        eval_losses = _stack([res[f"{method}_eval_loss_final"] for res in per_seed])
        eval_histories = _stack([res[f"{method}_eval_loss_history"] for res in per_seed])
        train_histories = _stack([res[f"{method}_train_loss_history"] for res in per_seed])
        theta_histories = _stack([res[f"{method}_theta_history"] for res in per_seed])
        history_steps = np.asarray(per_seed[0][f"{method}_history_steps"], dtype=np.int32)

        results.update(
            {
                f"{method}_theta_finals": theta_finals,
                f"{method}_theta_mean": np.mean(theta_finals, axis=0),
                f"{method}_theta_std": np.std(theta_finals, axis=0),
                f"{method}_theta_se": np.std(theta_finals, axis=0) / np.sqrt(max(len(theta_finals), 1)),
                f"{method}_eval_losses": eval_losses,
                f"{method}_eval_mean": np.mean(eval_losses),
                f"{method}_eval_std": np.std(eval_losses),
                f"{method}_eval_se": np.std(eval_losses) / np.sqrt(max(len(eval_losses), 1)),
                f"{method}_history_steps": history_steps,
                f"{method}_eval_histories": eval_histories,
                f"{method}_train_histories": train_histories,
                f"{method}_theta_histories": theta_histories,
                f"{method}_eval_history_mean": np.mean(eval_histories, axis=0),
                f"{method}_train_history_mean": np.mean(train_histories, axis=0),
                f"{method}_theta_history_mean": np.mean(theta_histories, axis=0),
            }
        )

    results.update(
        {
            "m_obs": np.asarray(kwargs.get("m_obs", 100), dtype=np.int32),
            "n_model": np.asarray(kwargs.get("n_model", 50), dtype=np.int32),
            "n_steps": np.asarray(kwargs.get("n_steps", 500), dtype=np.int32),
            "sgd_n_steps": np.asarray(kwargs.get("sgd_n_steps", kwargs.get("n_steps", 500)), dtype=np.int32),
            "pgd_n_steps": np.asarray(kwargs.get("pgd_n_steps", kwargs.get("n_steps", 500)), dtype=np.int32),
            "num_steps": np.asarray(kwargs.get("num_steps", 100), dtype=np.int32),
            "T": np.asarray(kwargs.get("T", 1.0), dtype=np.float64),
            "ell_fixed": np.asarray(kwargs.get("ell_fixed", 30.0), dtype=np.float64),
            "ell_eval": np.asarray(kwargs.get("ell_eval", 30.0), dtype=np.float64),
            "sgd_gamma": np.asarray(kwargs.get("sgd_gamma", 1e-2), dtype=np.float64),
            "pgd_ell0": np.asarray(kwargs.get("pgd_ell0", 100.0), dtype=np.float64),
            "pgd_ell_min": np.asarray(kwargs.get("pgd_ell_min", 30.0), dtype=np.float64),
            "pgd_decay": np.asarray(kwargs.get("pgd_decay", 0.995), dtype=np.float64),
            "pgd_gamma": np.asarray(kwargs.get("pgd_gamma", 1e-2), dtype=np.float64),
            "scale_eps": np.asarray(kwargs.get("scale_eps", 1e-6), dtype=np.float64),
        }
    )

    if output_path is not None:
        save_results(results, output_path)
    return results


def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **results)


def load_results(input_path):
    with np.load(input_path) as data:
        return {key: data[key] for key in data.files}


if __name__ == "__main__":
    # SGD MMD uses a fixed RBF lengthscale.
    # PGD starts from pgd_ell0 and decays down to pgd_ell_min.
    standardize = False
    ell_fixed = 30.0
    ell_eval = 30.0
    pgd_ell0 = 1000.0
    pgd_ell_min = 30.0
    pgd_decay = 0.9995
    sgd_gamma = 100
    pgd_gamma = 100
    pgd_lambda_scale = 1e-3
    sgd_n_steps = 12000 
    pgd_n_steps = 12000

    result = run_experiment(
        seeds=range(10),
        output_path="lotka_volterra_results_90_90_new.npz",
        corruption=0.0,
        n_steps=max(sgd_n_steps, pgd_n_steps),
        sgd_n_steps=sgd_n_steps,
        pgd_n_steps=pgd_n_steps,
        num_steps=50,
        standardize=standardize,
        ell_fixed=ell_fixed,
        ell_eval=ell_eval,
        pgd_ell0=pgd_ell0,
        pgd_ell_min=pgd_ell_min,
        pgd_decay=pgd_decay,
        run_plain_sgd=True,
        sgd_gamma=sgd_gamma,
        pgd_gamma=pgd_gamma,
        pgd_lambda_scale=pgd_lambda_scale,
        history_every=1,
        print_every=10,
    )
    print("SGD MMD theta mean:", result["sgd_theta_mean"])
    print("PGD theta mean:", result["pgd_theta_mean"])
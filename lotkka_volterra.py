from functools import partial
from pathlib import Path
import time

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
    history_every=100,
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
    target_batch_size=100,
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
    history_every=100,
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
        key, key_batch, key_model, key_eval = jax.random.split(key, 4)
        y_batch = _sample_target_batch(key_batch, y_obs_full, target_batch_size)
        ell_t = float(ell_schedule[t])
        gamma_t = jnp.asarray(gamma, dtype=jnp.float64)
        lambda_t = jnp.asarray(lambda_scale, dtype=jnp.float64)
        phi, train_loss, direction, theta_delta = pgd_step_phi(
            phi,
            key_model,
            y_batch,
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
def _pairwise_distance_quantiles(samples, quantiles=(0.5, 0.9), max_pairs=20000, seed=0):
    samples = np.asarray(samples, dtype=np.float64)
    n = samples.shape[0]
    if n < 2:
        return np.full((len(quantiles),), np.nan, dtype=np.float64)

    rng = np.random.default_rng(seed)
    max_unique_pairs = n * (n - 1) // 2
    pair_count = int(min(max_pairs, max_unique_pairs))
    i = rng.integers(0, n, size=pair_count)
    j = rng.integers(0, n, size=pair_count)
    same = i == j
    while np.any(same):
        j[same] = rng.integers(0, n, size=int(np.sum(same)))
        same = i == j

    distances = np.linalg.norm(samples[i] - samples[j], axis=1)
    return np.asarray(np.quantile(distances, quantiles), dtype=np.float64)


def _cross_distance_quantiles(x, y, quantiles=(0.5, 0.9), max_pairs=20000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] == 0 or y.shape[0] == 0:
        return np.full((len(quantiles),), np.nan, dtype=np.float64)

    rng = np.random.default_rng(seed)
    pair_count = int(min(max_pairs, x.shape[0] * y.shape[0]))
    i = rng.integers(0, x.shape[0], size=pair_count)
    j = rng.integers(0, y.shape[0], size=pair_count)
    distances = np.linalg.norm(x[i] - y[j], axis=1)
    return np.asarray(np.quantile(distances, quantiles), dtype=np.float64)


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
    history_every=100,
    print_every=100,
    run_plain_sgd=True,
    sgd_gamma=1e-2,
    standardize=False,
    scale_eps=1e-6,
    print_kernel_diagnostics=False,
    kernel_diag_max_pairs=20000,
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
    y_obs_scaled = (y_obs_full - scale_mean) / scale_std
    dist_q50, dist_q90 = _pairwise_distance_quantiles(
        y_obs_scaled,
        quantiles=(0.5, 0.9),
        max_pairs=kernel_diag_max_pairs,
        seed=seed + 777,
    )
    key_diag = jax.random.PRNGKey(seed + 5555)
    noises_diag = jax.random.normal(
        key_diag,
        shape=(n_eval_model, num_steps, 3),
        dtype=jnp.float64,
    )
    x_diag = simulate_lv_samples_from_noises(theta0, noises_diag, T=T, theta1=theta1)
    x_diag_scaled = (x_diag - scale_mean) / scale_std
    cross_q50, cross_q90 = _cross_distance_quantiles(
        x_diag_scaled,
        y_obs_scaled,
        quantiles=(0.5, 0.9),
        max_pairs=kernel_diag_max_pairs,
        seed=seed + 778,
    )
    if print_kernel_diagnostics:
        suggested_lo = max(1e-12, 0.5 * float(dist_q50))
        suggested_hi = max(suggested_lo, 2.0 * float(dist_q90))
        ell_min_ratio = float(pgd_ell_min) / max(float(dist_q50), 1e-12)
        ell_eval_ratio = float(ell_eval) / max(float(cross_q50), 1e-12)
        print(
            f"[LV kernel diag] seed={seed:d} "
            f"obs_obs_q50={float(dist_q50):.6g} obs_obs_q90={float(dist_q90):.6g} "
            f"model_obs_q50={float(cross_q50):.6g} model_obs_q90={float(cross_q90):.6g} "
            f"suggested_ell~[{suggested_lo:.6g}, {suggested_hi:.6g}] "
            f"| ell_min/obs_q50={ell_min_ratio:.3f} "
            f"| ell_eval/model_obs_q50={ell_eval_ratio:.3f}"
        )

    result = {
        "theta_true": theta_true,
        "theta_bad": theta_bad,
        "theta0": theta0,
        "corruption": np.asarray(corruption, dtype=np.float64),
        "standardize": np.asarray(standardize, dtype=np.bool_),
        "scale_mean": np.asarray(scale_mean, dtype=np.float64),
        "scale_std": np.asarray(scale_std, dtype=np.float64),
        "kernel_diag_dist_q50": np.asarray(dist_q50, dtype=np.float64),
        "kernel_diag_dist_q90": np.asarray(dist_q90, dtype=np.float64),
        "kernel_diag_cross_q50": np.asarray(cross_q50, dtype=np.float64),
        "kernel_diag_cross_q90": np.asarray(cross_q90, dtype=np.float64),
    }

    if run_plain_sgd:
        sgd_start = time.perf_counter()
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
        sgd_elapsed_seconds = time.perf_counter() - sgd_start
        result.update(_prefix_result("sgd", sgd_res))
        result["sgd_elapsed_seconds"] = np.asarray(sgd_elapsed_seconds, dtype=np.float64)

    pgd_start = time.perf_counter()
    pgd_res = run_pgd(
        seed=seed,
        theta0=theta0,
        y_obs_full=y_obs_full,
        scale_mean=scale_mean,
        scale_std=scale_std,
        n_model=n_model,
        target_batch_size=target_batch_size,
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
    pgd_elapsed_seconds = time.perf_counter() - pgd_start
    result.update(_prefix_result("pgd", pgd_res))
    result["pgd_elapsed_seconds"] = np.asarray(pgd_elapsed_seconds, dtype=np.float64)
    return result


def _prefix_result(prefix, result):
    return {f"{prefix}_{key}": value for key, value in result.items()}


def _stack(values, dtype=np.float64):
    return np.asarray(values, dtype=dtype)


def _format_value_for_filename(value):
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value).replace("-", "m").replace(".", "p")


def _resolve_theta0_for_seed(seed, seed_index, theta0, theta0_by_seed):
    if theta0_by_seed is None:
        return theta0

    if isinstance(theta0_by_seed, dict):
        if seed not in theta0_by_seed:
            raise KeyError(f"Missing theta0 for seed {seed}.")
        return theta0_by_seed[seed]

    theta0_array = np.asarray(theta0_by_seed, dtype=np.float64)
    if theta0_array.ndim != 2:
        raise ValueError("theta0_by_seed must be a dict or a 2D array-like of shape (n_seeds, theta_dim).")
    if seed_index >= theta0_array.shape[0]:
        raise ValueError("theta0_by_seed has fewer rows than the number of seeds.")
    return theta0_array[seed_index]


def run_experiment(
    seeds=range(5),
    output_path=None,
    theta0_by_seed=None,
    **kwargs,
):
    seeds = list(seeds)
    per_seed = []
    theta0_per_seed = []

    for seed_index, seed in enumerate(seeds):
        print(f"\n=== Lotka-Volterra seed {seed} ===")
        theta0_seed = _resolve_theta0_for_seed(
            seed=seed,
            seed_index=seed_index,
            theta0=kwargs.get("theta0", np.array([90.0, 90.0], dtype=np.float64)),
            theta0_by_seed=theta0_by_seed,
        )
        theta0_per_seed.append(np.asarray(theta0_seed, dtype=np.float64))
        seed_kwargs = dict(kwargs)
        seed_kwargs["theta0"] = theta0_seed
        per_seed.append(run_one_seed(seed=seed, **seed_kwargs))

    method_names = ["pgd"]
    if any("sgd_theta_final" in res for res in per_seed):
        method_names.insert(0, "sgd")

    results = {
        "seeds": np.asarray(seeds, dtype=np.int32),
        "theta_true": per_seed[0]["theta_true"],
        "theta_bad": per_seed[0]["theta_bad"],
        "theta0": per_seed[0]["theta0"],
        "theta0_per_seed": _stack(theta0_per_seed),
        "corruption": per_seed[0]["corruption"],
        "standardize": per_seed[0]["standardize"],
        "scale_mean": per_seed[0]["scale_mean"],
        "scale_std": per_seed[0]["scale_std"],
        "kernel_diag_dist_q50": _stack([res["kernel_diag_dist_q50"] for res in per_seed]),
        "kernel_diag_dist_q90": _stack([res["kernel_diag_dist_q90"] for res in per_seed]),
        "kernel_diag_cross_q50": _stack([res["kernel_diag_cross_q50"] for res in per_seed]),
        "kernel_diag_cross_q90": _stack([res["kernel_diag_cross_q90"] for res in per_seed]),
    }

    for method in method_names:
        theta_finals = _stack([res[f"{method}_theta_final"] for res in per_seed])
        eval_losses = _stack([res[f"{method}_eval_loss_final"] for res in per_seed])
        eval_histories = _stack([res[f"{method}_eval_loss_history"] for res in per_seed])
        train_histories = _stack([res[f"{method}_train_loss_history"] for res in per_seed])
        theta_histories = _stack([res[f"{method}_theta_history"] for res in per_seed])
        history_steps = np.asarray(per_seed[0][f"{method}_history_steps"], dtype=np.int32)
        elapsed_seconds = _stack([res[f"{method}_elapsed_seconds"] for res in per_seed])

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
                f"{method}_elapsed_seconds": elapsed_seconds,
                f"{method}_elapsed_mean": np.mean(elapsed_seconds),
                f"{method}_elapsed_std": np.std(elapsed_seconds),
                f"{method}_elapsed_se": np.std(elapsed_seconds)
                / np.sqrt(max(len(elapsed_seconds), 1)),
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
            "kernel_diag_max_pairs": np.asarray(kwargs.get("kernel_diag_max_pairs", 20000), dtype=np.int32),
            "print_kernel_diagnostics": np.asarray(kwargs.get("print_kernel_diagnostics", False), dtype=np.bool_),
            "uses_theta0_by_seed": np.asarray(theta0_by_seed is not None, dtype=np.bool_),
        }
    )

    if output_path is not None:
        save_results(results, output_path)
    return results


def run_ablation_sweep(
    sweep_name,
    sweep_param,
    sweep_values,
    output_dir,
    seeds=range(5),
    file_prefix="lotka_volterra",
    **kwargs,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_values = list(sweep_values)
    output_paths = []
    per_value_results = []

    for sweep_value in sweep_values:
        run_kwargs = dict(kwargs)
        run_kwargs[sweep_param] = sweep_value
        value_tag = _format_value_for_filename(sweep_value)
        output_path = output_dir / f"{file_prefix}_{sweep_name}_{value_tag}.npz"
        result = run_experiment(
            seeds=seeds,
            output_path=output_path,
            **run_kwargs,
        )
        output_paths.append(str(output_path))
        per_value_results.append(result)

    summary = {
        "sweep_name": np.asarray(sweep_name),
        "sweep_param": np.asarray(sweep_param),
        "sweep_values": np.asarray(sweep_values, dtype=np.float64),
        "output_paths": np.asarray(output_paths, dtype=str),
        "seeds": np.asarray(list(seeds), dtype=np.int32),
    }

    for method in ("sgd", "pgd"):
        key = f"{method}_theta_mean"
        if key not in per_value_results[0]:
            continue
        theta_means = np.asarray([res[f"{method}_theta_mean"] for res in per_value_results], dtype=np.float64)
        theta_stds = np.asarray([res[f"{method}_theta_std"] for res in per_value_results], dtype=np.float64)
        eval_means = np.asarray([res[f"{method}_eval_mean"] for res in per_value_results], dtype=np.float64)
        eval_stds = np.asarray([res[f"{method}_eval_std"] for res in per_value_results], dtype=np.float64)
        summary.update(
            {
                f"{method}_theta_means": theta_means,
                f"{method}_theta_stds": theta_stds,
                f"{method}_eval_means": eval_means,
                f"{method}_eval_stds": eval_stds,
            }
        )

    summary_path = output_dir / f"{file_prefix}_{sweep_name}_summary.npz"
    save_results(summary, summary_path)
    summary["summary_path"] = str(summary_path)
    return summary


def run_lengthscale_ablation(
    output_dir,
    sweep_param,
    sweep_values,
    seeds=range(5),
    file_prefix="lotka_volterra_lengthscale_ablation",
    **kwargs,
):
    run_kwargs = dict(kwargs)
    run_kwargs.setdefault("run_plain_sgd", False)
    return run_ablation_sweep(
        sweep_name=f"{sweep_param}_sweep",
        sweep_param=sweep_param,
        sweep_values=sweep_values,
        output_dir=output_dir,
        seeds=seeds,
        file_prefix=file_prefix,
        **run_kwargs,
    )


def run_regularization_ablation(
    output_dir,
    lambda_scales,
    seeds=range(5),
    file_prefix="lotka_volterra_regularization_ablation",
    **kwargs,
):
    run_kwargs = dict(kwargs)
    run_kwargs.setdefault("run_plain_sgd", False)
    return run_ablation_sweep(
        sweep_name="pgd_lambda_scale_sweep",
        sweep_param="pgd_lambda_scale",
        sweep_values=lambda_scales,
        output_dir=output_dir,
        seeds=seeds,
        file_prefix=file_prefix,
        **run_kwargs,
    )


def run_lengthscale_regularization_grid(
    output_dir,
    lengthscale_param,
    lengthscale_values,
    lambda_scales,
    seeds=range(5),
    file_prefix="lotka_volterra_lengthscale_regularization_grid",
    **kwargs,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_kwargs = dict(kwargs)
    run_kwargs.setdefault("run_plain_sgd", False)

    lengthscale_values = list(lengthscale_values)
    lambda_scales = list(lambda_scales)

    eval_mean_grid = np.full((len(lambda_scales), len(lengthscale_values)), np.nan, dtype=np.float64)
    eval_std_grid = np.full((len(lambda_scales), len(lengthscale_values)), np.nan, dtype=np.float64)
    theta_mean_grid = np.full((len(lambda_scales), len(lengthscale_values), 2), np.nan, dtype=np.float64)
    theta_std_grid = np.full((len(lambda_scales), len(lengthscale_values), 2), np.nan, dtype=np.float64)
    output_paths = np.empty((len(lambda_scales), len(lengthscale_values)), dtype=object)

    for lambda_idx, lambda_scale in enumerate(lambda_scales):
        for ell_idx, lengthscale_value in enumerate(lengthscale_values):
            cell_kwargs = dict(run_kwargs)
            cell_kwargs["pgd_lambda_scale"] = lambda_scale
            cell_kwargs[lengthscale_param] = lengthscale_value
            # Keep reporting bandwidth aligned with the swept minimum lengthscale.
            if lengthscale_param == "pgd_ell_min":
                cell_kwargs["ell_eval"] = lengthscale_value
            output_path = output_dir / (
                f"{file_prefix}_{lengthscale_param}_{_format_value_for_filename(lengthscale_value)}"
                f"_lambda_{_format_value_for_filename(lambda_scale)}.npz"
            )
            result = run_experiment(
                seeds=seeds,
                output_path=output_path,
                **cell_kwargs,
            )
            eval_mean_grid[lambda_idx, ell_idx] = result["pgd_eval_mean"]
            eval_std_grid[lambda_idx, ell_idx] = result["pgd_eval_std"]
            theta_mean_grid[lambda_idx, ell_idx] = result["pgd_theta_mean"]
            theta_std_grid[lambda_idx, ell_idx] = result["pgd_theta_std"]
            output_paths[lambda_idx, ell_idx] = str(output_path)

    summary = {
        "grid_name": np.asarray("lengthscale_regularization_grid"),
        "lengthscale_param": np.asarray(lengthscale_param),
        "lengthscale_values": np.asarray(lengthscale_values, dtype=np.float64),
        "lambda_scales": np.asarray(lambda_scales, dtype=np.float64),
        "pgd_eval_mean_grid": eval_mean_grid,
        "pgd_eval_std_grid": eval_std_grid,
        "pgd_theta_mean_grid": theta_mean_grid,
        "pgd_theta_std_grid": theta_std_grid,
        "output_paths": np.asarray(output_paths, dtype=str),
        "seeds": np.asarray(list(seeds), dtype=np.int32),
    }

    summary_path = output_dir / f"{file_prefix}_{lengthscale_param}_summary.npz"
    save_results(summary, summary_path)
    summary["summary_path"] = str(summary_path)
    return summary


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
    ell_fixed = 30
    ell_eval = 30
    pgd_ell0 = 1000
    pgd_ell_min =30
    pgd_decay = 0.9995
    sgd_gamma = 100
    pgd_gamma = 100
    pgd_lambda_scale = 1e-3
    sgd_n_steps = 12000
    pgd_n_steps = 12000
    theta0_by_seed = np.tile(np.array([60.0, 60.0], dtype=np.float64), (10, 1))

    result = run_experiment(
        seeds=range(10),
        output_path="lotka_volterra_results_60_60_12000_c35.npz",
        corruption=0.35,
        n_steps=max(sgd_n_steps, pgd_n_steps),
        sgd_n_steps=sgd_n_steps,
        pgd_n_steps=pgd_n_steps,
        theta0_by_seed=theta0_by_seed,
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
        print_kernel_diagnostics=True,
        history_every=1,
        print_every=100,
    )
    print("SGD MMD theta mean:", result["sgd_theta_mean"])
    print("PGD theta mean:", result["pgd_theta_mean"])

    # summary = run_lengthscale_regularization_grid(
    #     output_dir="ablations/lv_heatmap",
    #     lengthscale_param="pgd_ell_min",
    #     lengthscale_values=[10, 30, 100, 300],
    #     lambda_scales=[1e-4, 1e-2, 1e-0],

    #     pgd_ell0=1000.0,   # fixed here
    #     pgd_decay=0.9995,
    #     pgd_gamma=100,
    #     ell_fixed=30.0,
    #     ell_eval=30.0,
    #     corruption=0.0,
    #     theta0_by_seed=np.array(
    #         [
    #             [50.0, 60.0],
    #             [90.0, 90.0],
    #             [75.0, 75.0],
    #             [55.0, 55.0],
    #             [60.0, 60.0],
    #         ],
    #         dtype=np.float64,
    #     ),
    #     m_obs=100,
    #     n_model=50,
    #     sgd_n_steps=12000,
    #     pgd_n_steps=15000,
    #     num_steps=50,
    #     T=1.0,
    #     standardize=False,
    #     seeds=range(5),
    # )

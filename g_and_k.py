import numpy as np
import time
from functools import partial
from pathlib import Path
from results_io import save_metadata_sidecar

import jax
jax.config.update("jax_enable_x64", True)
import jax.nn as jnn
import jax.numpy as jnp
from jax.scipy.special import ndtri


# ============================================================
# 1. Model: g-and-k quantile and Jacobian
# ============================================================
# This section defines the simulator x = Q(u; theta) and the manual
# derivative dQ/dtheta. Both optimizers use the same model and Jacobian.
def gk_quantile(u, theta):
    """Scalar g-and-k quantile for theta = [a, b, c, k]."""
    a, b, c, k = theta
    z = ndtri(u)

    exp_term = jnp.exp(-c * z)
    skew_core = (1.0 - exp_term) / (1.0 + exp_term)
    skew = 1.0 + 0.8 * skew_core

    log_base = jnp.log1p(z**2)
    tail = jnp.exp(k * log_base)

    return a + b * skew * tail * z


gk_quantile_vmap = jax.vmap(gk_quantile, in_axes=(0, None))


def gk_jac_theta_manual(u, theta):
    """Jacobian wrt theta = [a, b, c, k], returned as shape (4,)."""
    _, b, c, k = theta
    z = ndtri(u)

    exp_term = jnp.exp(-c * z)
    denom = 1.0 + exp_term

    skew_core = (1.0 - exp_term) / denom
    skew = 1.0 + 0.8 * skew_core

    log_base = jnp.log1p(z**2)
    tail = jnp.exp(k * log_base)
    common = tail * z

    da = 1.0
    db = skew * common
    dskew_dc = 1.6 * z * exp_term / (denom**2)
    dc = b * dskew_dc * common
    dk = b * skew * common * log_base

    return jnp.array([da, db, dc, dk], dtype=jnp.float64)


gk_jac_theta_vmap = jax.vmap(gk_jac_theta_manual, in_axes=(0, None))


def sample_gk(key, theta, n):
    """Draw model samples by sampling uniforms and applying the quantile."""
    eps = 1e-6
    u = jax.random.uniform(key, shape=(n,), minval=eps, maxval=1.0 - eps)
    x = gk_quantile_vmap(u, theta)
    return u, x


# ============================================================
# 2. MMD objective and witness gradient
# ============================================================
# The training loss and eval loss are MMD^2 values. The witness gradient
# gives d(MMD^2)/dx for the model particles, which is then chained through
# the g-and-k Jacobian to get parameter updates.
def gaussian_kernel_1d(x, y, ell):
    diff = x[:, None] - y[None, :]
    return jnp.exp(-0.5 * (diff**2) / (ell**2))


def mmd2_vstat_1d(x, y, ell):
    n = x.shape[0]
    m = y.shape[0]

    k_xx = gaussian_kernel_1d(x, x, ell)
    k_yy = gaussian_kernel_1d(y, y, ell)
    k_xy = gaussian_kernel_1d(x, y, ell)

    term_xx = jnp.sum(k_xx) / (n * n)
    term_yy = jnp.sum(k_yy) / (m * m)
    term_xy = jnp.sum(k_xy) / (n * m)

    return term_xx + term_yy - 2.0 * term_xy


def witness_gradient_empirical(x, y, ell):
    """Empirical witness gradient wrt model samples x, shape (n_model,)."""
    diff_xx = x[:, None] - x[None, :]
    k_xx = jnp.exp(-0.5 * (diff_xx**2) / (ell**2))
    grad_emp = jnp.mean(k_xx * (-diff_xx) / (ell**2), axis=1)

    diff_yx = y[:, None] - x[None, :]
    k_yx = jnp.exp(-0.5 * (diff_yx**2) / (ell**2))
    grad_tar = jnp.mean(k_yx * (diff_yx) / (ell**2), axis=0)

    return grad_emp - grad_tar


@jax.jit
def lhs_rhs_values_gk(x_model, y_target, ell_t, ell_inf):
    data_dim = 1.0
    grad = witness_gradient_empirical(x_model, y_target, ell_inf)
    scale = ((ell_t**2) / (ell_inf**2)) ** (0.5 * data_dim)
    lhs = scale * jnp.mean(grad**2)
    rhs = mmd2_vstat_1d(x_model, y_target, ell_inf)
    return lhs, rhs


def make_adaptive_ell_schedule(n_steps, ell0, ell_min, decay):
    """Geometric decay schedule clipped below by ell_min."""
    ts = np.arange(n_steps, dtype=np.float64)
    return np.maximum(ell_min, ell0 * (decay ** ts))


# ============================================================
# 3. Parameter transform: theta <-> phi
# ============================================================
# Optimization happens in phi-space. Parameters 1, 3, and 4 use centered
# range scaling. Parameter 2 is represented through a softplus latent so
# that b stays positive.
def theta_min():
    return jnp.array([2.0, 0.3, 0.75, -0.8], dtype=jnp.float64)


def theta_max():
    return jnp.array([4.0, 1.8, 1.5, -0.4], dtype=jnp.float64)


def theta_center():
    return 0.5 * (theta_min() + theta_max())


def theta_width():
    return 0.5 * (theta_max() - theta_min())


def phi_to_theta(phi):
    center = theta_center()
    width = theta_width()

    a = center[0] + width[0] * phi[0]

    # b is positive, so phi[1] maps to an unconstrained latent b_raw.
    b_raw = center[1] + width[1] * phi[1]
    b = jnn.softplus(b_raw) + 1e-6

    c = center[2] + width[2] * phi[2]
    k = center[3] + width[3] * phi[3]

    return jnp.array([a, b, c, k], dtype=jnp.float64)


def theta_to_phi(theta):
    center = theta_center()
    width = theta_width()

    a_phi = (theta[0] - center[0]) / width[0]

    # Inverse softplus for b: if b = softplus(b_raw) + eps,
    # then b_raw = log(exp(b - eps) - 1).
    b_raw = jnp.log(jnp.expm1(jnp.maximum(theta[1] - 1e-6, 1e-12)))
    b_phi = (b_raw - center[1]) / width[1]

    c_phi = (theta[2] - center[2]) / width[2]
    k_phi = (theta[3] - center[3]) / width[3]

    return jnp.array([a_phi, b_phi, c_phi, k_phi], dtype=jnp.float64)


# ============================================================
# 4. One-step optimizers
# ============================================================
@partial(jax.jit, static_argnums=(2,))
def eval_loss_full(theta, key_eval, n_eval_model, y_obs_full, ell_t):
    _, x_eval = sample_gk(key_eval, theta, n_eval_model)
    return mmd2_vstat_1d(x_eval, y_obs_full, ell_t)


@partial(jax.jit, static_argnums=(3,))
def gd_step_phi(phi, key_model, y_batch, n_model, ell_t, gamma_t):
    theta = phi_to_theta(phi)
    u_model, x_model = sample_gk(key_model, theta, n_model)

    grad_f = witness_gradient_empirical(x_model, y_batch, ell_t)
    J_theta = gk_jac_theta_vmap(u_model, theta)
    jac_col_norms = jnp.sqrt(jnp.mean(J_theta**2, axis=0))
    grad_theta = jnp.mean(J_theta * grad_f[:, None], axis=0)

    grad_phi = grad_theta * theta_width()
    phi_new = phi - gamma_t * grad_phi
    theta_delta = phi_to_theta(phi_new) - theta

    train_loss = mmd2_vstat_1d(x_model, y_batch, ell_t)
    return phi_new, train_loss, grad_theta, theta_delta, jac_col_norms


@partial(jax.jit, static_argnums=(3,))
def natural_step_phi(phi, key_model, y_batch, n_model, ell_t, gamma_t, damping_t):
    theta = phi_to_theta(phi)
    u_model, x_model = sample_gk(key_model, theta, n_model)

    grad_f = witness_gradient_empirical(x_model, y_batch, ell_t)
    J_theta = gk_jac_theta_vmap(u_model, theta)
    jac_col_norms = jnp.sqrt(jnp.mean(J_theta**2, axis=0))
    J_phi = J_theta * theta_width()[None, :]

    grad_phi = jnp.mean(J_phi * grad_f[:, None], axis=0)
    fisher_phi = (J_phi.T @ J_phi) / n_model + damping_t * jnp.eye(phi.shape[0], dtype=phi.dtype)
    direction = jnp.linalg.solve(fisher_phi, grad_phi)

    phi_new = phi - gamma_t * direction
    theta_delta = phi_to_theta(phi_new) - theta

    train_loss = mmd2_vstat_1d(x_model, y_batch, ell_t)
    return phi_new, train_loss, grad_phi, direction, theta_delta, jac_col_norms


@partial(jax.jit, static_argnums=(3,))
def pgd_step_phi(phi, key_model, y_batch, n_model, ell_t, gamma_t, lambda_t):
    theta = phi_to_theta(phi)
    u_model, x_model = sample_gk(key_model, theta, n_model)

    grad_f = witness_gradient_empirical(x_model, y_batch, ell_t)
    J_theta = gk_jac_theta_vmap(u_model, theta)
    jac_col_norms = jnp.sqrt(jnp.mean(J_theta**2, axis=0))
    J_phi = J_theta * theta_width()[None, :]

    A = (J_phi.T @ J_phi) / n_model + lambda_t * jnp.eye(phi.shape[0], dtype=phi.dtype)
    direction = jnp.mean(J_phi * grad_f[:, None], axis=0)
    delta = jnp.linalg.solve(A, direction)

    phi_new = phi - gamma_t * delta
    theta_delta = phi_to_theta(phi_new) - theta

    train_loss = mmd2_vstat_1d(x_model, y_batch, ell_t)
    return phi_new, train_loss, direction, theta_delta, jac_col_norms


# ============================================================
# 5. Data generation and single-method runners
# ============================================================
# make_target_and_init creates the synthetic observed data for one seed.
# run_baseline_sgd and run_natural_sgd use target minibatches. run_adaptive_pgd
# uses the full observed target for PGD updates, which reduces target-side noise.
def make_target_and_init(seed, theta_true, n_obs_full, theta0=None):
    key = jax.random.PRNGKey(seed)
    key, key_obs = jax.random.split(key)
    _, y_obs_full = sample_gk(key_obs, theta_true, n_obs_full)

    if theta0 is None:
        theta0 = jnp.array([2.0, 2.0, 1.5, -0.3], dtype=jnp.float64)
    else:
        theta0 = jnp.array(theta0, dtype=jnp.float64)

    return y_obs_full, theta0


def run_baseline_sgd(
    seed,
    theta0,
    y_obs_full,
    target_batch_size,
    n_model,
    n_steps_sgd,
    gamma_sgd,
    ell_fixed,
    ell_eval,
    ell_schedule=None,
    n_eval_model=2000,
    print_every=20,
    history_every=1,
    method_label="SGD",
):
    key = jax.random.PRNGKey(seed + 1000)
    phi = theta_to_phi(theta0)

    history_steps = []
    train_loss_history = []
    eval_loss_history = []
    theta_history = []
    grad_theta_history = []
    theta_delta_history = []
    jac_col_norm_history = []
    last_train_loss = None
    last_eval_loss = None

    for t in range(n_steps_sgd):
        key, key_batch, key_model, key_eval = jax.random.split(key, 4)

        idx = jax.random.randint(
            key_batch,
            shape=(target_batch_size,),
            minval=0,
            maxval=y_obs_full.shape[0],
        )
        y_batch = y_obs_full[idx]
        ell_value = ell_fixed if ell_schedule is None else ell_schedule[t]
        ell_t = jnp.asarray(ell_value, dtype=jnp.float64)
        ell_eval_t = jnp.asarray(ell_eval, dtype=jnp.float64)

        phi, train_loss, grad_theta, theta_delta, jac_col_norms = gd_step_phi(
            phi=phi,
            key_model=key_model,
            y_batch=y_batch,
            n_model=n_model,
            ell_t=ell_t,
            gamma_t=jnp.asarray(gamma_sgd, dtype=jnp.float64),
        )
        last_train_loss = train_loss

        if (t % history_every == 0) or (t == n_steps_sgd - 1):
            theta = phi_to_theta(phi)
            eval_loss = eval_loss_full(
                theta=theta,
                key_eval=key_eval,
                n_eval_model=n_eval_model,
                y_obs_full=y_obs_full,
                ell_t=ell_eval_t,
            )
            last_eval_loss = eval_loss
            history_steps.append(t)
            train_loss_history.append(float(train_loss))
            eval_loss_history.append(float(eval_loss))
            theta_history.append(np.array(theta, dtype=np.float64))
            grad_theta_history.append(np.array(grad_theta, dtype=np.float64))
            theta_delta_history.append(np.array(theta_delta, dtype=np.float64))
            jac_col_norm_history.append(np.array(jac_col_norms, dtype=np.float64))

        if (t % print_every == 0) or (t == n_steps_sgd - 1):
            if last_eval_loss is None:
                theta = phi_to_theta(phi)
                last_eval_loss = eval_loss_full(
                    theta=theta,
                    key_eval=key_eval,
                    n_eval_model=n_eval_model,
                    y_obs_full=y_obs_full,
                    ell_t=ell_eval_t,
                )

            print(
                f"[{method_label}] step={t:4d} | ell={float(ell_t):.4f} | "
                f"gamma={float(gamma_sgd):.6f} | "
                f"train_loss={float(train_loss):.8f} | "
                f"eval_loss={float(last_eval_loss):.8f}"
            )

    return {
        "theta_final": np.array(phi_to_theta(phi), dtype=np.float64),
        "train_loss_final": float(last_train_loss),
        "eval_loss_final": float(last_eval_loss) if last_eval_loss is not None else None,
        "history_steps": np.array(history_steps, dtype=np.int32),
        "train_loss_history": np.array(train_loss_history, dtype=np.float64),
        "eval_loss_history": np.array(eval_loss_history, dtype=np.float64),
        "theta_history": np.array(theta_history, dtype=np.float64),
        "grad_theta_history": np.array(grad_theta_history, dtype=np.float64),
        "theta_delta_history": np.array(theta_delta_history, dtype=np.float64),
        "jac_col_norm_history": np.array(jac_col_norm_history, dtype=np.float64),
    }


def run_natural_sgd(
    seed,
    theta0,
    y_obs_full,
    target_batch_size,
    n_model,
    n_steps_sgd,
    gamma_natural_sgd,
    natural_damping,
    ell_fixed,
    ell_eval,
    n_eval_model=2000,
    print_every=20,
    history_every=1,
):
    key = jax.random.PRNGKey(seed + 1500)
    phi = theta_to_phi(theta0)

    history_steps = []
    train_loss_history = []
    eval_loss_history = []
    theta_history = []
    grad_phi_history = []
    direction_history = []
    theta_delta_history = []
    jac_col_norm_history = []
    last_train_loss = None
    last_eval_loss = None

    for t in range(n_steps_sgd):
        key, key_batch, key_model, key_eval = jax.random.split(key, 4)

        idx = jax.random.randint(
            key_batch,
            shape=(target_batch_size,),
            minval=0,
            maxval=y_obs_full.shape[0],
        )
        y_batch = y_obs_full[idx]
        ell_t = jnp.asarray(ell_fixed, dtype=jnp.float64)
        ell_eval_t = jnp.asarray(ell_eval, dtype=jnp.float64)

        phi, train_loss, grad_phi, direction, theta_delta, jac_col_norms = natural_step_phi(
            phi=phi,
            key_model=key_model,
            y_batch=y_batch,
            n_model=n_model,
            ell_t=ell_t,
            gamma_t=jnp.asarray(gamma_natural_sgd, dtype=jnp.float64),
            damping_t=jnp.asarray(natural_damping, dtype=jnp.float64),
        )
        last_train_loss = train_loss

        if (t % history_every == 0) or (t == n_steps_sgd - 1):
            theta = phi_to_theta(phi)
            eval_loss = eval_loss_full(
                theta=theta,
                key_eval=key_eval,
                n_eval_model=n_eval_model,
                y_obs_full=y_obs_full,
                ell_t=ell_eval_t,
            )
            last_eval_loss = eval_loss
            history_steps.append(t)
            train_loss_history.append(float(train_loss))
            eval_loss_history.append(float(eval_loss))
            theta_history.append(np.array(theta, dtype=np.float64))
            grad_phi_history.append(np.array(grad_phi, dtype=np.float64))
            direction_history.append(np.array(direction, dtype=np.float64))
            theta_delta_history.append(np.array(theta_delta, dtype=np.float64))
            jac_col_norm_history.append(np.array(jac_col_norms, dtype=np.float64))

        if (t % print_every == 0) or (t == n_steps_sgd - 1):
            if last_eval_loss is None:
                theta = phi_to_theta(phi)
                last_eval_loss = eval_loss_full(
                    theta=theta,
                    key_eval=key_eval,
                    n_eval_model=n_eval_model,
                    y_obs_full=y_obs_full,
                    ell_t=ell_eval_t,
                )

            print(
                f"[Natural SGD] step={t:4d} | ell={float(ell_t):.4f} | "
                f"gamma={float(gamma_natural_sgd):.6f} | damping={float(natural_damping):.6f} | "
                f"train_loss={float(train_loss):.8f} | eval_loss={float(last_eval_loss):.8f}"
            )

    return {
        "theta_final": np.array(phi_to_theta(phi), dtype=np.float64),
        "train_loss_final": float(last_train_loss),
        "eval_loss_final": float(last_eval_loss) if last_eval_loss is not None else None,
        "history_steps": np.array(history_steps, dtype=np.int32),
        "train_loss_history": np.array(train_loss_history, dtype=np.float64),
        "eval_loss_history": np.array(eval_loss_history, dtype=np.float64),
        "theta_history": np.array(theta_history, dtype=np.float64),
        "grad_phi_history": np.array(grad_phi_history, dtype=np.float64),
        "direction_history": np.array(direction_history, dtype=np.float64),
        "theta_delta_history": np.array(theta_delta_history, dtype=np.float64),
        "jac_col_norm_history": np.array(jac_col_norm_history, dtype=np.float64),
    }


def run_adaptive_pgd(
    seed,
    theta0,
    y_obs_full,
    target_batch_size,
    n_model,
    n_steps_pgd,
    gamma_pgd0,
    lambda_scale,
    ell0,
    ell_min,
    decay,
    ell_eval,
    ell_schedule=None,
    n_eval_model=2000,
    print_every=20,
    history_every=1,
    method_label="PGD",
):
    key = jax.random.PRNGKey(seed + 2000)
    phi = theta_to_phi(theta0)

    history_steps = []
    train_loss_history = []
    eval_loss_history = []
    theta_history = []
    direction_history = []
    theta_delta_history = []
    jac_col_norm_history = []
    lhs_history = []
    rhs_history = []
    last_train_loss = None
    last_eval_loss = None

    if ell_schedule is None:
        ell_schedule = make_adaptive_ell_schedule(n_steps_pgd, ell0, ell_min, decay)
    else:
        ell_schedule = np.asarray(ell_schedule, dtype=np.float64)
        if ell_schedule.shape[0] != n_steps_pgd:
            raise ValueError("ell_schedule must have length n_steps_pgd.")

    for t in range(n_steps_pgd):
        key, key_batch, key_model, key_eval = jax.random.split(key, 4)
        if target_batch_size is None or target_batch_size >= y_obs_full.shape[0]:
            y_batch = y_obs_full
        else:
            idx = jax.random.randint(
                key_batch,
                shape=(target_batch_size,),
                minval=0,
                maxval=y_obs_full.shape[0],
            )
            y_batch = y_obs_full[idx]

        ell_t = jnp.asarray(ell_schedule[t], dtype=jnp.float64)
        ell_eval_t = jnp.asarray(ell_eval, dtype=jnp.float64)
        gamma_t = jnp.asarray(gamma_pgd0, dtype=jnp.float64)
        lambda_t = jnp.asarray(lambda_scale, dtype=jnp.float64)

        phi, train_loss, direction, theta_delta, jac_col_norms = pgd_step_phi(
            phi=phi,
            key_model=key_model,
            y_batch=y_batch,
            n_model=n_model,
            ell_t=ell_t,
            gamma_t=gamma_t,
            lambda_t=lambda_t,
        )
        last_train_loss = train_loss

        if (t % history_every == 0) or (t == n_steps_pgd - 1):
            theta = phi_to_theta(phi)
            eval_loss = eval_loss_full(
                theta=theta,
                key_eval=key_eval,
                n_eval_model=n_eval_model,
                y_obs_full=y_obs_full,
                ell_t=ell_eval_t,
            )
            last_eval_loss = eval_loss
            history_steps.append(t)
            train_loss_history.append(float(train_loss))
            eval_loss_history.append(float(eval_loss))
            theta_history.append(np.array(theta, dtype=np.float64))
            direction_history.append(np.array(direction, dtype=np.float64))
            theta_delta_history.append(np.array(theta_delta, dtype=np.float64))
            jac_col_norm_history.append(np.array(jac_col_norms, dtype=np.float64))
            # Optional diagnostics, disabled for fairer runtime comparisons:
            # _, x_diag = sample_gk(key_eval, theta, n_model)
            # lhs, rhs = lhs_rhs_values_gk(
            #     x_model=x_diag,
            #     y_target=y_obs_full,
            #     ell_t=ell_t,
            #     ell_inf=jnp.asarray(ell_min, dtype=jnp.float64),
            # )
            # lhs_history.append(float(lhs))
            # rhs_history.append(float(rhs))
            # lhs_history.append(np.nan)
            # rhs_history.append(np.nan)

        if (t % print_every == 0) or (t == n_steps_pgd - 1):
            # Optional diagnostics, disabled for fairer runtime comparisons:
            # theta = phi_to_theta(phi)
            # _, x_diag = sample_gk(key_eval, theta, n_model)
            # lhs, rhs = lhs_rhs_values_gk(
            #     x_model=x_diag,
            #     y_target=y_obs_full,
            #     ell_t=ell_t,
            #     ell_inf=jnp.asarray(ell_min, dtype=jnp.float64),
            # )
            # ratio = float(lhs) / float(rhs) if float(rhs) != 0.0 else np.inf
            if last_eval_loss is None:
                theta = phi_to_theta(phi)
                last_eval_loss = eval_loss_full(
                    theta=theta,
                    key_eval=key_eval,
                    n_eval_model=n_eval_model,
                    y_obs_full=y_obs_full,
                    ell_t=ell_eval_t,
                )

            print(
                f"[{method_label}] step={t:4d} | ell={float(ell_t):.4f} | "
                f"gamma={float(gamma_t):.6f} | lambda={float(lambda_t):.6f} | "
                f"train_loss={float(train_loss):.8f} | "
                f"eval_loss={float(last_eval_loss):.8f}"
                # f" | lhs={float(lhs):.6e} | rhs={float(rhs):.6e} | "
                # f"lhs/rhs={ratio:.6e}"
            )

    return {
        "theta_final": np.array(phi_to_theta(phi), dtype=np.float64),
        "train_loss_final": float(last_train_loss),
        "eval_loss_final": float(last_eval_loss) if last_eval_loss is not None else None,
        "history_steps": np.array(history_steps, dtype=np.int32),
        "train_loss_history": np.array(train_loss_history, dtype=np.float64),
        "eval_loss_history": np.array(eval_loss_history, dtype=np.float64),
        "theta_history": np.array(theta_history, dtype=np.float64),
        "direction_history": np.array(direction_history, dtype=np.float64),
        "theta_delta_history": np.array(theta_delta_history, dtype=np.float64),
        "jac_col_norm_history": np.array(jac_col_norm_history, dtype=np.float64),
        "lhs_history": np.array(lhs_history, dtype=np.float64),
        "rhs_history": np.array(rhs_history, dtype=np.float64),
    }


# ============================================================
# 6. Combined one-seed experiment
# ============================================================
# This function creates one observed dataset, then runs GD and PGD from the
# same theta0 on that same dataset. It returns a flat dict that is easy to
# aggregate across seeds.
def run_baseline_and_adaptive(
    seed=0,
    theta_true=np.array([3.0, 1.0, 1.0, -np.log(2.0)], dtype=np.float64),
    theta0=None,
    n_obs_full=500,
    target_batch_size=200,
    n_model=300,
    n_steps_sgd=5000,
    n_steps_pgd=2000,
    gamma_sgd=0.01,
    gamma_natural_sgd=0.01,
    gamma_pgd0=0.49,
    lambda_scale=1e-3,
    natural_damping=1e-3,
    n_eval_model=2000,
    ell_fixed=2.0,
    ell0=20.0,
    ell_min=2.0,
    decay=0.997,
    history_every=1,
    run_baseline=True,
    run_natural=True,
    run_adaptive_sgd=False,
    run_fixed_pgd=False,
):
    theta_true = jnp.array(theta_true, dtype=jnp.float64)
    y_obs_full, theta0 = make_target_and_init(seed, theta_true, n_obs_full, theta0=theta0)
    gamma_natural_sgd = gamma_sgd if gamma_natural_sgd is None else gamma_natural_sgd
    natural_damping = lambda_scale if natural_damping is None else natural_damping

    result = {}
    adaptive_ell_schedule_sgd = make_adaptive_ell_schedule(n_steps_sgd, ell0, ell_min, decay)
    fixed_ell_schedule_pgd = np.full((n_steps_pgd,), ell_min, dtype=np.float64)

    if run_baseline:
        print("\n=== Plain SGD baseline (fixed lengthscale, no preconditioning) ===")
        baseline_start = time.perf_counter()
        baseline_res = run_baseline_sgd(
            seed=seed,
            theta0=theta0,
            y_obs_full=y_obs_full,
            target_batch_size=target_batch_size,
            n_model=n_model,
            n_steps_sgd=n_steps_sgd,
            gamma_sgd=gamma_sgd,
            ell_fixed=ell_fixed,
            ell_eval=ell_min,
            n_eval_model=n_eval_model,
            print_every=20,
            history_every=history_every,
        )
        baseline_elapsed_seconds = time.perf_counter() - baseline_start
        result.update(
            {
                "baseline_theta_final": baseline_res["theta_final"],
                "baseline_train_loss_final": baseline_res["train_loss_final"],
                "baseline_eval_loss_final": baseline_res["eval_loss_final"],
                "baseline_history_steps": baseline_res["history_steps"],
                "baseline_train_loss_history": baseline_res["train_loss_history"],
                "baseline_eval_loss_history": baseline_res["eval_loss_history"],
                "baseline_theta_history": baseline_res["theta_history"],
                "baseline_grad_theta_history": baseline_res["grad_theta_history"],
                "baseline_theta_delta_history": baseline_res["theta_delta_history"],
                "baseline_jac_col_norm_history": baseline_res["jac_col_norm_history"],
                "baseline_elapsed_seconds": np.asarray(baseline_elapsed_seconds, dtype=np.float64),
            }
        )

    if run_adaptive_sgd:
        print("\n=== Plain SGD with adaptive lengthscale schedule ===")
        adaptive_sgd_start = time.perf_counter()
        adaptive_sgd_res = run_baseline_sgd(
            seed=seed,
            theta0=theta0,
            y_obs_full=y_obs_full,
            target_batch_size=target_batch_size,
            n_model=n_model,
            n_steps_sgd=n_steps_sgd,
            gamma_sgd=gamma_sgd,
            ell_fixed=ell_fixed,
            ell_eval=ell_min,
            ell_schedule=adaptive_ell_schedule_sgd,
            n_eval_model=n_eval_model,
            print_every=20,
            history_every=history_every,
            method_label="SGD-adaptive-ell",
        )
        adaptive_sgd_elapsed_seconds = time.perf_counter() - adaptive_sgd_start
        result.update(
            {
                "adaptive_sgd_theta_final": adaptive_sgd_res["theta_final"],
                "adaptive_sgd_train_loss_final": adaptive_sgd_res["train_loss_final"],
                "adaptive_sgd_eval_loss_final": adaptive_sgd_res["eval_loss_final"],
                "adaptive_sgd_history_steps": adaptive_sgd_res["history_steps"],
                "adaptive_sgd_train_loss_history": adaptive_sgd_res["train_loss_history"],
                "adaptive_sgd_eval_loss_history": adaptive_sgd_res["eval_loss_history"],
                "adaptive_sgd_theta_history": adaptive_sgd_res["theta_history"],
                "adaptive_sgd_grad_theta_history": adaptive_sgd_res["grad_theta_history"],
                "adaptive_sgd_theta_delta_history": adaptive_sgd_res["theta_delta_history"],
                "adaptive_sgd_jac_col_norm_history": adaptive_sgd_res["jac_col_norm_history"],
                "adaptive_sgd_elapsed_seconds": np.asarray(adaptive_sgd_elapsed_seconds, dtype=np.float64),
            }
        )

    if run_natural:
        print("\n=== Natural SGD baseline (fixed lengthscale, local preconditioning) ===")
        natural_start = time.perf_counter()
        natural_res = run_natural_sgd(
            seed=seed,
            theta0=theta0,
            y_obs_full=y_obs_full,
            target_batch_size=target_batch_size,
            n_model=n_model,
            n_steps_sgd=n_steps_sgd,
            gamma_natural_sgd=gamma_natural_sgd,
            natural_damping=natural_damping,
            ell_fixed=ell_fixed,
            ell_eval=ell_min,
            n_eval_model=n_eval_model,
            print_every=20,
            history_every=history_every,
        )
        natural_elapsed_seconds = time.perf_counter() - natural_start
        result.update(
            {
                "natural_theta_final": natural_res["theta_final"],
                "natural_train_loss_final": natural_res["train_loss_final"],
                "natural_eval_loss_final": natural_res["eval_loss_final"],
                "natural_history_steps": natural_res["history_steps"],
                "natural_train_loss_history": natural_res["train_loss_history"],
                "natural_eval_loss_history": natural_res["eval_loss_history"],
                "natural_theta_history": natural_res["theta_history"],
                "natural_grad_phi_history": natural_res["grad_phi_history"],
                "natural_direction_history": natural_res["direction_history"],
                "natural_theta_delta_history": natural_res["theta_delta_history"],
                "natural_jac_col_norm_history": natural_res["jac_col_norm_history"],
                "natural_elapsed_seconds": np.asarray(natural_elapsed_seconds, dtype=np.float64),
            }
        )

    print("\n=== Adaptive PGD (shrinking lengthscale with local preconditioning) ===")
    adaptive_start = time.perf_counter()
    adaptive_res = run_adaptive_pgd(
        seed=seed,
        theta0=theta0,
        y_obs_full=y_obs_full,
        target_batch_size=target_batch_size,
        n_model=n_model,
        n_steps_pgd=n_steps_pgd,
        gamma_pgd0=gamma_pgd0,
        lambda_scale=lambda_scale,
        ell0=ell0,
        ell_min=ell_min,
        decay=decay,
        ell_eval=ell_min,
        n_eval_model=n_eval_model,
        print_every=20,
        history_every=history_every,
    )
    adaptive_elapsed_seconds = time.perf_counter() - adaptive_start
    result.update(
        {
            "adaptive_theta_final": adaptive_res["theta_final"],
            "adaptive_train_loss_final": adaptive_res["train_loss_final"],
            "adaptive_eval_loss_final": adaptive_res["eval_loss_final"],
            "adaptive_history_steps": adaptive_res["history_steps"],
            "adaptive_train_loss_history": adaptive_res["train_loss_history"],
            "adaptive_eval_loss_history": adaptive_res["eval_loss_history"],
            "adaptive_theta_history": adaptive_res["theta_history"],
            "adaptive_direction_history": adaptive_res["direction_history"],
            "adaptive_theta_delta_history": adaptive_res["theta_delta_history"],
            "adaptive_jac_col_norm_history": adaptive_res["jac_col_norm_history"],
            "adaptive_lhs_history": adaptive_res["lhs_history"],
            "adaptive_rhs_history": adaptive_res["rhs_history"],
            "adaptive_elapsed_seconds": np.asarray(adaptive_elapsed_seconds, dtype=np.float64),
        }
    )

    if run_fixed_pgd:
        print("\n=== PGD with fixed lengthscale ell_infty ===")
        fixed_pgd_start = time.perf_counter()
        fixed_pgd_res = run_adaptive_pgd(
            seed=seed,
            theta0=theta0,
            y_obs_full=y_obs_full,
            target_batch_size=target_batch_size,
            n_model=n_model,
            n_steps_pgd=n_steps_pgd,
            gamma_pgd0=gamma_pgd0,
            lambda_scale=lambda_scale,
            ell0=ell0,
            ell_min=ell_min,
            decay=decay,
            ell_eval=ell_min,
            ell_schedule=fixed_ell_schedule_pgd,
            n_eval_model=n_eval_model,
            print_every=20,
            history_every=history_every,
            method_label="PGD-fixed-ell",
        )
        fixed_pgd_elapsed_seconds = time.perf_counter() - fixed_pgd_start
        result.update(
            {
                "fixed_pgd_theta_final": fixed_pgd_res["theta_final"],
                "fixed_pgd_train_loss_final": fixed_pgd_res["train_loss_final"],
                "fixed_pgd_eval_loss_final": fixed_pgd_res["eval_loss_final"],
                "fixed_pgd_history_steps": fixed_pgd_res["history_steps"],
                "fixed_pgd_train_loss_history": fixed_pgd_res["train_loss_history"],
                "fixed_pgd_eval_loss_history": fixed_pgd_res["eval_loss_history"],
                "fixed_pgd_theta_history": fixed_pgd_res["theta_history"],
                "fixed_pgd_direction_history": fixed_pgd_res["direction_history"],
                "fixed_pgd_theta_delta_history": fixed_pgd_res["theta_delta_history"],
                "fixed_pgd_jac_col_norm_history": fixed_pgd_res["jac_col_norm_history"],
                "fixed_pgd_lhs_history": fixed_pgd_res["lhs_history"],
                "fixed_pgd_rhs_history": fixed_pgd_res["rhs_history"],
                "fixed_pgd_elapsed_seconds": np.asarray(fixed_pgd_elapsed_seconds, dtype=np.float64),
            }
        )

    return result


# ============================================================
# 7. Multi-seed aggregation and saving
# ============================================================
# This section repeats the one-seed experiment, stacks all histories, saves
# per-seed arrays, and also saves means/stds for quick plotting.
def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **results)
    save_metadata_sidecar(results, output_path)


def _format_theta_for_filename(theta):
    parts = []
    for value in np.asarray(theta, dtype=np.float64):
        token = f"{value:.3f}".replace("-", "m").replace(".", "p")
        parts.append(token)
    return "_".join(parts)


def _stack(values, dtype=np.float64):
    return np.array(values, dtype=dtype)


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


def run_grid_over_n_model(
    seeds,
    n_models,
    output_dir,
    theta_true=np.array([3.0, 1.0, 1.0, -np.log(2.0)], dtype=np.float64),
    theta0=None,
    theta0_by_seed=None,
    n_obs_full=500,
    target_batch_size=200,
    n_steps_sgd=5000,
    n_steps_pgd=2000,
    gamma_sgd=0.01,
    gamma_natural_sgd=0.01,
    gamma_pgd0=0.49,
    lambda_scale=1e-3,
    natural_damping=1e-3,
    n_eval_model=2000,
    ell_fixed=2.0,
    ell0=20.0,
    ell_min=2.0,
    decay=0.997,
    history_every=1,
    run_baseline=True,
    run_natural=True,
    run_adaptive_sgd=False,
    run_fixed_pgd=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds)

    theta0_value = (
        np.array([2.0, 2.0, 1.5, -0.3], dtype=np.float64)
        if theta0 is None
        else np.array(theta0, dtype=np.float64)
    )
    if theta0_value.ndim > 1:
        theta0_value = np.asarray(theta0_value[0], dtype=np.float64)
    theta0_tag = _format_theta_for_filename(theta0_value)
    gamma_natural_sgd_value = gamma_sgd if gamma_natural_sgd is None else gamma_natural_sgd
    natural_damping_value = lambda_scale if natural_damping is None else natural_damping
    summary = {}

    for n_model in n_models:
        per_seed = []
        theta0_per_seed = []
        for seed_index, seed in enumerate(seeds):
            theta0_seed = np.asarray(
                _resolve_theta0_for_seed(
                    seed=seed,
                    seed_index=seed_index,
                    theta0=theta0_value,
                    theta0_by_seed=theta0_by_seed,
                ),
                dtype=np.float64,
            )
            theta0_per_seed.append(theta0_seed)
            per_seed.append(
                run_baseline_and_adaptive(
                    seed=seed,
                    theta_true=theta_true,
                    theta0=theta0_seed,
                    n_obs_full=n_obs_full,
                    target_batch_size=target_batch_size,
                    n_model=n_model,
                    n_steps_sgd=n_steps_sgd,
                    n_steps_pgd=n_steps_pgd,
                    gamma_sgd=gamma_sgd,
                    gamma_natural_sgd=gamma_natural_sgd,
                    gamma_pgd0=gamma_pgd0,
                    lambda_scale=lambda_scale,
                    natural_damping=natural_damping,
                    n_eval_model=n_eval_model,
                    ell_fixed=ell_fixed,
                    ell0=ell0,
                    ell_min=ell_min,
                    decay=decay,
                    history_every=history_every,
                    run_baseline=run_baseline,
                    run_natural=run_natural,
                    run_adaptive_sgd=run_adaptive_sgd,
                    run_fixed_pgd=run_fixed_pgd,
                )
            )

        adaptive_train = _stack([res["adaptive_train_loss_final"] for res in per_seed])
        adaptive_eval = _stack([res["adaptive_eval_loss_final"] for res in per_seed])
        adaptive_thetas = _stack([res["adaptive_theta_final"] for res in per_seed])

        adaptive_history_steps = np.array(per_seed[0]["adaptive_history_steps"], dtype=np.int32)

        adaptive_train_histories = _stack([res["adaptive_train_loss_history"] for res in per_seed])
        adaptive_eval_histories = _stack([res["adaptive_eval_loss_history"] for res in per_seed])
        adaptive_theta_histories = _stack([res["adaptive_theta_history"] for res in per_seed])
        adaptive_direction_histories = _stack([res["adaptive_direction_history"] for res in per_seed])
        adaptive_theta_delta_histories = _stack([res["adaptive_theta_delta_history"] for res in per_seed])
        adaptive_jac_col_norm_histories = _stack([res["adaptive_jac_col_norm_history"] for res in per_seed])
        adaptive_lhs_histories = _stack([res["adaptive_lhs_history"] for res in per_seed])
        adaptive_rhs_histories = _stack([res["adaptive_rhs_history"] for res in per_seed])
        adaptive_elapsed_seconds = _stack([res["adaptive_elapsed_seconds"] for res in per_seed])

        results = {
            "seeds": np.array(seeds, dtype=np.int32),
            "n_model": np.array(n_model, dtype=np.int32),
            "theta_true": np.array(theta_true, dtype=np.float64),
            "theta0": theta0_value,
            "theta0_per_seed": _stack(theta0_per_seed),
            "n_obs_full": np.array(n_obs_full, dtype=np.int32),
            "target_batch_size": np.array(target_batch_size, dtype=np.int32),
            "n_steps_sgd": np.array(n_steps_sgd, dtype=np.int32),
            "n_steps_pgd": np.array(n_steps_pgd, dtype=np.int32),
            "gamma_sgd": np.array(gamma_sgd, dtype=np.float64),
            "gamma_natural_sgd": np.array(gamma_natural_sgd, dtype=np.float64),
            "gamma_pgd0": np.array(gamma_pgd0, dtype=np.float64),
            "lambda_scale": np.array(lambda_scale, dtype=np.float64),
            "natural_damping": np.array(natural_damping, dtype=np.float64),
            "n_eval_model": np.array(n_eval_model, dtype=np.int32),
            "ell_fixed": np.array(ell_fixed, dtype=np.float64),
            "ell0": np.array(ell0, dtype=np.float64),
            "ell_min": np.array(ell_min, dtype=np.float64),
            "ell_eval": np.array(ell_min, dtype=np.float64),
            "decay": np.array(decay, dtype=np.float64),
            "history_every": np.array(history_every, dtype=np.int32),
            "adaptive_train_losses": adaptive_train,
            "adaptive_eval_losses": adaptive_eval,
            "adaptive_thetas": adaptive_thetas,
            "adaptive_theta_mean": np.mean(adaptive_thetas, axis=0),
            "adaptive_theta_std": np.std(adaptive_thetas, axis=0),
            "adaptive_train_mean": np.mean(adaptive_train),
            "adaptive_train_std": np.std(adaptive_train),
            "adaptive_eval_mean": np.mean(adaptive_eval),
            "adaptive_eval_std": np.std(adaptive_eval),
            "adaptive_elapsed_seconds": adaptive_elapsed_seconds,
            "adaptive_elapsed_mean": np.mean(adaptive_elapsed_seconds),
            "adaptive_elapsed_std": np.std(adaptive_elapsed_seconds),
            "adaptive_elapsed_se": np.std(adaptive_elapsed_seconds) / np.sqrt(max(len(adaptive_elapsed_seconds), 1)),
            "adaptive_history_steps": adaptive_history_steps,
            "adaptive_train_histories": adaptive_train_histories,
            "adaptive_eval_histories": adaptive_eval_histories,
            "adaptive_theta_histories": adaptive_theta_histories,
            "adaptive_direction_histories": adaptive_direction_histories,
            "adaptive_theta_delta_histories": adaptive_theta_delta_histories,
            "adaptive_jac_col_norm_histories": adaptive_jac_col_norm_histories,
            "adaptive_lhs_histories": adaptive_lhs_histories,
            "adaptive_rhs_histories": adaptive_rhs_histories,
            "adaptive_train_history_mean": np.mean(adaptive_train_histories, axis=0),
            "adaptive_eval_history_mean": np.mean(adaptive_eval_histories, axis=0),
            "adaptive_theta_history_mean": np.mean(adaptive_theta_histories, axis=0),
            "adaptive_direction_history_mean": np.mean(adaptive_direction_histories, axis=0),
            "adaptive_theta_delta_history_mean": np.mean(adaptive_theta_delta_histories, axis=0),
            "adaptive_jac_col_norm_history_mean": np.mean(adaptive_jac_col_norm_histories, axis=0),
            "adaptive_lhs_history_mean": np.mean(adaptive_lhs_histories, axis=0),
            "adaptive_rhs_history_mean": np.mean(adaptive_rhs_histories, axis=0),
            "last_adapt_checkpoint_steps": adaptive_history_steps,
            "last_adapt_lhs": adaptive_lhs_histories[-1],
            "last_adapt_rhs": adaptive_rhs_histories[-1],
            "run_baseline": np.array(run_baseline, dtype=np.bool_),
            "run_natural": np.array(run_natural, dtype=np.bool_),
            "run_adaptive_sgd": np.array(run_adaptive_sgd, dtype=np.bool_),
            "run_fixed_pgd": np.array(run_fixed_pgd, dtype=np.bool_),
            "uses_theta0_by_seed": np.array(theta0_by_seed is not None, dtype=np.bool_),
        }

        if run_baseline:
            baseline_train = _stack([res["baseline_train_loss_final"] for res in per_seed])
            baseline_eval = _stack([res["baseline_eval_loss_final"] for res in per_seed])
            baseline_thetas = _stack([res["baseline_theta_final"] for res in per_seed])
            baseline_history_steps = np.array(per_seed[0]["baseline_history_steps"], dtype=np.int32)
            baseline_train_histories = _stack([res["baseline_train_loss_history"] for res in per_seed])
            baseline_eval_histories = _stack([res["baseline_eval_loss_history"] for res in per_seed])
            baseline_theta_histories = _stack([res["baseline_theta_history"] for res in per_seed])
            baseline_grad_theta_histories = _stack([res["baseline_grad_theta_history"] for res in per_seed])
            baseline_theta_delta_histories = _stack([res["baseline_theta_delta_history"] for res in per_seed])
            baseline_jac_col_norm_histories = _stack([res["baseline_jac_col_norm_history"] for res in per_seed])
            baseline_elapsed_seconds = _stack([res["baseline_elapsed_seconds"] for res in per_seed])

            results.update(
                {
                    "baseline_train_losses": baseline_train,
                    "baseline_eval_losses": baseline_eval,
                    "baseline_thetas": baseline_thetas,
                    "baseline_theta_mean": np.mean(baseline_thetas, axis=0),
                    "baseline_theta_std": np.std(baseline_thetas, axis=0),
                    "baseline_train_mean": np.mean(baseline_train),
                    "baseline_train_std": np.std(baseline_train),
                    "baseline_eval_mean": np.mean(baseline_eval),
                    "baseline_eval_std": np.std(baseline_eval),
                    "baseline_elapsed_seconds": baseline_elapsed_seconds,
                    "baseline_elapsed_mean": np.mean(baseline_elapsed_seconds),
                    "baseline_elapsed_std": np.std(baseline_elapsed_seconds),
                    "baseline_elapsed_se": np.std(baseline_elapsed_seconds)
                    / np.sqrt(max(len(baseline_elapsed_seconds), 1)),
                    "baseline_history_steps": baseline_history_steps,
                    "baseline_train_histories": baseline_train_histories,
                    "baseline_eval_histories": baseline_eval_histories,
                    "baseline_theta_histories": baseline_theta_histories,
                    "baseline_grad_theta_histories": baseline_grad_theta_histories,
                    "baseline_theta_delta_histories": baseline_theta_delta_histories,
                    "baseline_jac_col_norm_histories": baseline_jac_col_norm_histories,
                    "baseline_train_history_mean": np.mean(baseline_train_histories, axis=0),
                    "baseline_eval_history_mean": np.mean(baseline_eval_histories, axis=0),
                    "baseline_theta_history_mean": np.mean(baseline_theta_histories, axis=0),
                    "baseline_grad_theta_history_mean": np.mean(baseline_grad_theta_histories, axis=0),
                    "baseline_theta_delta_history_mean": np.mean(baseline_theta_delta_histories, axis=0),
                    "baseline_jac_col_norm_history_mean": np.mean(baseline_jac_col_norm_histories, axis=0),
                }
            )

        if run_natural:
            natural_train = _stack([res["natural_train_loss_final"] for res in per_seed])
            natural_eval = _stack([res["natural_eval_loss_final"] for res in per_seed])
            natural_thetas = _stack([res["natural_theta_final"] for res in per_seed])
            natural_history_steps = np.array(per_seed[0]["natural_history_steps"], dtype=np.int32)
            natural_train_histories = _stack([res["natural_train_loss_history"] for res in per_seed])
            natural_eval_histories = _stack([res["natural_eval_loss_history"] for res in per_seed])
            natural_theta_histories = _stack([res["natural_theta_history"] for res in per_seed])
            natural_grad_phi_histories = _stack([res["natural_grad_phi_history"] for res in per_seed])
            natural_direction_histories = _stack([res["natural_direction_history"] for res in per_seed])
            natural_theta_delta_histories = _stack([res["natural_theta_delta_history"] for res in per_seed])
            natural_jac_col_norm_histories = _stack([res["natural_jac_col_norm_history"] for res in per_seed])
            natural_elapsed_seconds = _stack([res["natural_elapsed_seconds"] for res in per_seed])

            results.update(
                {
                    "natural_train_losses": natural_train,
                    "natural_eval_losses": natural_eval,
                    "natural_thetas": natural_thetas,
                    "natural_theta_mean": np.mean(natural_thetas, axis=0),
                    "natural_theta_std": np.std(natural_thetas, axis=0),
                    "natural_train_mean": np.mean(natural_train),
                    "natural_train_std": np.std(natural_train),
                    "natural_eval_mean": np.mean(natural_eval),
                    "natural_eval_std": np.std(natural_eval),
                    "natural_elapsed_seconds": natural_elapsed_seconds,
                    "natural_elapsed_mean": np.mean(natural_elapsed_seconds),
                    "natural_elapsed_std": np.std(natural_elapsed_seconds),
                    "natural_elapsed_se": np.std(natural_elapsed_seconds)
                    / np.sqrt(max(len(natural_elapsed_seconds), 1)),
                    "natural_history_steps": natural_history_steps,
                    "natural_train_histories": natural_train_histories,
                    "natural_eval_histories": natural_eval_histories,
                    "natural_theta_histories": natural_theta_histories,
                    "natural_grad_phi_histories": natural_grad_phi_histories,
                    "natural_direction_histories": natural_direction_histories,
                    "natural_theta_delta_histories": natural_theta_delta_histories,
                    "natural_jac_col_norm_histories": natural_jac_col_norm_histories,
                    "natural_train_history_mean": np.mean(natural_train_histories, axis=0),
                    "natural_eval_history_mean": np.mean(natural_eval_histories, axis=0),
                    "natural_theta_history_mean": np.mean(natural_theta_histories, axis=0),
                    "natural_grad_phi_history_mean": np.mean(natural_grad_phi_histories, axis=0),
                    "natural_direction_history_mean": np.mean(natural_direction_histories, axis=0),
                    "natural_theta_delta_history_mean": np.mean(natural_theta_delta_histories, axis=0),
                    "natural_jac_col_norm_history_mean": np.mean(natural_jac_col_norm_histories, axis=0),
                }
            )

        if run_adaptive_sgd:
            adaptive_sgd_train = _stack([res["adaptive_sgd_train_loss_final"] for res in per_seed])
            adaptive_sgd_eval = _stack([res["adaptive_sgd_eval_loss_final"] for res in per_seed])
            adaptive_sgd_thetas = _stack([res["adaptive_sgd_theta_final"] for res in per_seed])
            adaptive_sgd_history_steps = np.array(per_seed[0]["adaptive_sgd_history_steps"], dtype=np.int32)
            adaptive_sgd_train_histories = _stack([res["adaptive_sgd_train_loss_history"] for res in per_seed])
            adaptive_sgd_eval_histories = _stack([res["adaptive_sgd_eval_loss_history"] for res in per_seed])
            adaptive_sgd_theta_histories = _stack([res["adaptive_sgd_theta_history"] for res in per_seed])
            adaptive_sgd_grad_theta_histories = _stack([res["adaptive_sgd_grad_theta_history"] for res in per_seed])
            adaptive_sgd_theta_delta_histories = _stack([res["adaptive_sgd_theta_delta_history"] for res in per_seed])
            adaptive_sgd_jac_col_norm_histories = _stack([res["adaptive_sgd_jac_col_norm_history"] for res in per_seed])
            adaptive_sgd_elapsed_seconds = _stack([res["adaptive_sgd_elapsed_seconds"] for res in per_seed])
            results.update(
                {
                    "adaptive_sgd_train_losses": adaptive_sgd_train,
                    "adaptive_sgd_eval_losses": adaptive_sgd_eval,
                    "adaptive_sgd_thetas": adaptive_sgd_thetas,
                    "adaptive_sgd_theta_mean": np.mean(adaptive_sgd_thetas, axis=0),
                    "adaptive_sgd_theta_std": np.std(adaptive_sgd_thetas, axis=0),
                    "adaptive_sgd_train_mean": np.mean(adaptive_sgd_train),
                    "adaptive_sgd_train_std": np.std(adaptive_sgd_train),
                    "adaptive_sgd_eval_mean": np.mean(adaptive_sgd_eval),
                    "adaptive_sgd_eval_std": np.std(adaptive_sgd_eval),
                    "adaptive_sgd_elapsed_seconds": adaptive_sgd_elapsed_seconds,
                    "adaptive_sgd_elapsed_mean": np.mean(adaptive_sgd_elapsed_seconds),
                    "adaptive_sgd_elapsed_std": np.std(adaptive_sgd_elapsed_seconds),
                    "adaptive_sgd_elapsed_se": np.std(adaptive_sgd_elapsed_seconds)
                    / np.sqrt(max(len(adaptive_sgd_elapsed_seconds), 1)),
                    "adaptive_sgd_history_steps": adaptive_sgd_history_steps,
                    "adaptive_sgd_train_histories": adaptive_sgd_train_histories,
                    "adaptive_sgd_eval_histories": adaptive_sgd_eval_histories,
                    "adaptive_sgd_theta_histories": adaptive_sgd_theta_histories,
                    "adaptive_sgd_grad_theta_histories": adaptive_sgd_grad_theta_histories,
                    "adaptive_sgd_theta_delta_histories": adaptive_sgd_theta_delta_histories,
                    "adaptive_sgd_jac_col_norm_histories": adaptive_sgd_jac_col_norm_histories,
                    "adaptive_sgd_train_history_mean": np.mean(adaptive_sgd_train_histories, axis=0),
                    "adaptive_sgd_eval_history_mean": np.mean(adaptive_sgd_eval_histories, axis=0),
                    "adaptive_sgd_theta_history_mean": np.mean(adaptive_sgd_theta_histories, axis=0),
                    "adaptive_sgd_grad_theta_history_mean": np.mean(adaptive_sgd_grad_theta_histories, axis=0),
                    "adaptive_sgd_theta_delta_history_mean": np.mean(adaptive_sgd_theta_delta_histories, axis=0),
                    "adaptive_sgd_jac_col_norm_history_mean": np.mean(adaptive_sgd_jac_col_norm_histories, axis=0),
                }
            )

        if run_fixed_pgd:
            fixed_pgd_train = _stack([res["fixed_pgd_train_loss_final"] for res in per_seed])
            fixed_pgd_eval = _stack([res["fixed_pgd_eval_loss_final"] for res in per_seed])
            fixed_pgd_thetas = _stack([res["fixed_pgd_theta_final"] for res in per_seed])
            fixed_pgd_history_steps = np.array(per_seed[0]["fixed_pgd_history_steps"], dtype=np.int32)
            fixed_pgd_train_histories = _stack([res["fixed_pgd_train_loss_history"] for res in per_seed])
            fixed_pgd_eval_histories = _stack([res["fixed_pgd_eval_loss_history"] for res in per_seed])
            fixed_pgd_theta_histories = _stack([res["fixed_pgd_theta_history"] for res in per_seed])
            fixed_pgd_direction_histories = _stack([res["fixed_pgd_direction_history"] for res in per_seed])
            fixed_pgd_theta_delta_histories = _stack([res["fixed_pgd_theta_delta_history"] for res in per_seed])
            fixed_pgd_jac_col_norm_histories = _stack([res["fixed_pgd_jac_col_norm_history"] for res in per_seed])
            fixed_pgd_lhs_histories = _stack([res["fixed_pgd_lhs_history"] for res in per_seed])
            fixed_pgd_rhs_histories = _stack([res["fixed_pgd_rhs_history"] for res in per_seed])
            fixed_pgd_elapsed_seconds = _stack([res["fixed_pgd_elapsed_seconds"] for res in per_seed])
            results.update(
                {
                    "fixed_pgd_train_losses": fixed_pgd_train,
                    "fixed_pgd_eval_losses": fixed_pgd_eval,
                    "fixed_pgd_thetas": fixed_pgd_thetas,
                    "fixed_pgd_theta_mean": np.mean(fixed_pgd_thetas, axis=0),
                    "fixed_pgd_theta_std": np.std(fixed_pgd_thetas, axis=0),
                    "fixed_pgd_train_mean": np.mean(fixed_pgd_train),
                    "fixed_pgd_train_std": np.std(fixed_pgd_train),
                    "fixed_pgd_eval_mean": np.mean(fixed_pgd_eval),
                    "fixed_pgd_eval_std": np.std(fixed_pgd_eval),
                    "fixed_pgd_elapsed_seconds": fixed_pgd_elapsed_seconds,
                    "fixed_pgd_elapsed_mean": np.mean(fixed_pgd_elapsed_seconds),
                    "fixed_pgd_elapsed_std": np.std(fixed_pgd_elapsed_seconds),
                    "fixed_pgd_elapsed_se": np.std(fixed_pgd_elapsed_seconds)
                    / np.sqrt(max(len(fixed_pgd_elapsed_seconds), 1)),
                    "fixed_pgd_history_steps": fixed_pgd_history_steps,
                    "fixed_pgd_train_histories": fixed_pgd_train_histories,
                    "fixed_pgd_eval_histories": fixed_pgd_eval_histories,
                    "fixed_pgd_theta_histories": fixed_pgd_theta_histories,
                    "fixed_pgd_direction_histories": fixed_pgd_direction_histories,
                    "fixed_pgd_theta_delta_histories": fixed_pgd_theta_delta_histories,
                    "fixed_pgd_jac_col_norm_histories": fixed_pgd_jac_col_norm_histories,
                    "fixed_pgd_lhs_histories": fixed_pgd_lhs_histories,
                    "fixed_pgd_rhs_histories": fixed_pgd_rhs_histories,
                    "fixed_pgd_train_history_mean": np.mean(fixed_pgd_train_histories, axis=0),
                    "fixed_pgd_eval_history_mean": np.mean(fixed_pgd_eval_histories, axis=0),
                    "fixed_pgd_theta_history_mean": np.mean(fixed_pgd_theta_histories, axis=0),
                    "fixed_pgd_direction_history_mean": np.mean(fixed_pgd_direction_histories, axis=0),
                    "fixed_pgd_theta_delta_history_mean": np.mean(fixed_pgd_theta_delta_histories, axis=0),
                    "fixed_pgd_jac_col_norm_history_mean": np.mean(fixed_pgd_jac_col_norm_histories, axis=0),
                    "fixed_pgd_lhs_history_mean": np.mean(fixed_pgd_lhs_histories, axis=0),
                    "fixed_pgd_rhs_history_mean": np.mean(fixed_pgd_rhs_histories, axis=0),
                }
            )

        output_path = output_dir / f"g_n_k_fixed{n_model}_theta0_{theta0_tag}.npz"
        save_results(results, output_path)
        summary[int(n_model)] = {
            "output_path": str(output_path),
            "adaptive_eval_mean": float(results["adaptive_eval_mean"]),
            "adaptive_theta_mean": results["adaptive_theta_mean"].tolist(),
        }
        if run_baseline:
            summary[int(n_model)].update(
                {
                    "baseline_eval_mean": float(results["baseline_eval_mean"]),
                    "baseline_theta_mean": results["baseline_theta_mean"].tolist(),
                }
            )
        if run_natural:
            summary[int(n_model)].update(
                {
                    "natural_eval_mean": float(results["natural_eval_mean"]),
                    "natural_theta_mean": results["natural_theta_mean"].tolist(),
                }
            )
        if run_adaptive_sgd:
            summary[int(n_model)].update(
                {
                    "adaptive_sgd_eval_mean": float(results["adaptive_sgd_eval_mean"]),
                    "adaptive_sgd_theta_mean": results["adaptive_sgd_theta_mean"].tolist(),
                }
            )
        if run_fixed_pgd:
            summary[int(n_model)].update(
                {
                    "fixed_pgd_eval_mean": float(results["fixed_pgd_eval_mean"]),
                    "fixed_pgd_theta_mean": results["fixed_pgd_theta_mean"].tolist(),
                }
            )

    return summary


def run_for_n_model(
    n_model,
    seeds,
    output_dir,
    theta_true=np.array([3.0, 1.0, 1.0, -np.log(2.0)], dtype=np.float64),
    theta0=None,
    theta0_by_seed=None,
    n_obs_full=500,
    target_batch_size=200,
    n_steps_sgd=5000,
    n_steps_pgd=2000,
    gamma_sgd=0.01,
    gamma_natural_sgd=0.01,
    gamma_pgd0=0.49,
    lambda_scale=1e-3,
    natural_damping=1e-3,
    n_eval_model=2000,
    ell_fixed=2.0,
    ell0=20.0,
    ell_min=2.0,
    decay=0.997,
    history_every=1,
    run_baseline=True,
    run_natural=True,
    run_adaptive_sgd=False,
    run_fixed_pgd=False,
):
    summary = run_grid_over_n_model(
        seeds=seeds,
        n_models=[n_model],
        output_dir=output_dir,
        theta0=theta0,
        theta0_by_seed=theta0_by_seed,
        theta_true=theta_true,
        n_obs_full=n_obs_full,
        target_batch_size=target_batch_size,
        n_steps_sgd=n_steps_sgd,
        n_steps_pgd=n_steps_pgd,
        gamma_sgd=gamma_sgd,
        gamma_natural_sgd=gamma_natural_sgd,
        gamma_pgd0=gamma_pgd0,
        lambda_scale=lambda_scale,
        natural_damping=natural_damping,
        n_eval_model=n_eval_model,
        ell_fixed=ell_fixed,
        ell0=ell0,
        ell_min=ell_min,
        decay=decay,
        history_every=history_every,
        run_baseline=run_baseline,
        run_natural=run_natural,
        run_adaptive_sgd=run_adaptive_sgd,
        run_fixed_pgd=run_fixed_pgd,
    )
    return summary[int(n_model)]


def run_ablation_for_n_model(
    n_model,
    seeds,
    output_dir,
    sweep_name,
    sweep_param,
    sweep_values,
    file_prefix="g_and_k_ablation",
    **kwargs,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_values = list(sweep_values)
    output_paths = []
    per_value_summaries = []

    for sweep_value in sweep_values:
        run_kwargs = dict(kwargs)
        run_kwargs[sweep_param] = sweep_value
        sweep_output_dir = output_dir / f"{file_prefix}_{sweep_name}_{_format_value_for_filename(sweep_value)}"
        sweep_output_dir.mkdir(parents=True, exist_ok=True)
        summary = run_for_n_model(
            n_model=n_model,
            seeds=seeds,
            output_dir=sweep_output_dir,
            **run_kwargs,
        )
        output_paths.append(summary["output_path"])
        per_value_summaries.append(summary)

    aggregated = {
        "sweep_name": np.asarray(sweep_name),
        "sweep_param": np.asarray(sweep_param),
        "sweep_values": np.asarray(sweep_values, dtype=np.float64),
        "output_paths": np.asarray(output_paths, dtype=str),
        "n_model": np.asarray(n_model, dtype=np.int32),
        "seeds": np.asarray(list(seeds), dtype=np.int32),
        "adaptive_eval_means": np.asarray(
            [summary["adaptive_eval_mean"] for summary in per_value_summaries],
            dtype=np.float64,
        ),
        "adaptive_theta_means": np.asarray(
            [summary["adaptive_theta_mean"] for summary in per_value_summaries],
            dtype=np.float64,
        ),
    }

    if all("baseline_eval_mean" in summary for summary in per_value_summaries):
        aggregated["baseline_eval_means"] = np.asarray(
            [summary["baseline_eval_mean"] for summary in per_value_summaries],
            dtype=np.float64,
        )
        aggregated["baseline_theta_means"] = np.asarray(
            [summary["baseline_theta_mean"] for summary in per_value_summaries],
            dtype=np.float64,
        )

    if all("natural_eval_mean" in summary for summary in per_value_summaries):
        aggregated["natural_eval_means"] = np.asarray(
            [summary["natural_eval_mean"] for summary in per_value_summaries],
            dtype=np.float64,
        )
        aggregated["natural_theta_means"] = np.asarray(
            [summary["natural_theta_mean"] for summary in per_value_summaries],
            dtype=np.float64,
        )

    summary_path = output_dir / f"{file_prefix}_{sweep_name}_summary.npz"
    save_results(aggregated, summary_path)
    aggregated["summary_path"] = str(summary_path)
    return aggregated


def run_lengthscale_ablation_for_n_model(
    n_model,
    seeds,
    output_dir,
    sweep_param,
    sweep_values,
    file_prefix="g_and_k_lengthscale_ablation",
    **kwargs,
):
    run_kwargs = dict(kwargs)
    run_kwargs.setdefault("run_baseline", False)
    run_kwargs.setdefault("run_natural", False)
    return run_ablation_for_n_model(
        n_model=n_model,
        seeds=seeds,
        output_dir=output_dir,
        sweep_name=f"{sweep_param}_sweep",
        sweep_param=sweep_param,
        sweep_values=sweep_values,
        file_prefix=file_prefix,
        **run_kwargs,
    )


def run_regularization_ablation_for_n_model(
    n_model,
    seeds,
    output_dir,
    lambda_scales,
    file_prefix="g_and_k_regularization_ablation",
    **kwargs,
):
    run_kwargs = dict(kwargs)
    run_kwargs.setdefault("run_baseline", False)
    run_kwargs.setdefault("run_natural", False)
    return run_ablation_for_n_model(
        n_model=n_model,
        seeds=seeds,
        output_dir=output_dir,
        sweep_name="lambda_scale_sweep",
        sweep_param="lambda_scale",
        sweep_values=lambda_scales,
        file_prefix=file_prefix,
        **run_kwargs,
    )


def run_step_size_ablation_for_n_model(
    n_model,
    seeds,
    output_dir,
    gamma_values,
    sweep_param="gamma_pgd0",
    file_prefix="g_and_k_step_size_ablation",
    **kwargs,
):
    run_kwargs = dict(kwargs)
    run_kwargs.setdefault("run_baseline", False)
    run_kwargs.setdefault("run_natural", False)
    return run_ablation_for_n_model(
        n_model=n_model,
        seeds=seeds,
        output_dir=output_dir,
        sweep_name=f"{sweep_param}_sweep",
        sweep_param=sweep_param,
        sweep_values=gamma_values,
        file_prefix=file_prefix,
        **run_kwargs,
    )


def run_decay_ablation_for_n_model(
    n_model,
    seeds,
    output_dir,
    decay_values,
    sweep_param="decay",
    file_prefix="g_and_k_decay_ablation",
    **kwargs,
):
    run_kwargs = dict(kwargs)
    run_kwargs.setdefault("run_baseline", False)
    run_kwargs.setdefault("run_natural", False)
    return run_ablation_for_n_model(
        n_model=n_model,
        seeds=seeds,
        output_dir=output_dir,
        sweep_name=f"{sweep_param}_sweep",
        sweep_param=sweep_param,
        sweep_values=decay_values,
        file_prefix=file_prefix,
        **run_kwargs,
    )


def run_observation_model_grid(
    seeds,
    n_obs_full_values,
    n_model_values,
    output_dir,
    file_prefix="g_and_k_observation_model_grid",
    tie_target_batch_to_n_obs=True,
    **kwargs,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_obs_full_values = list(n_obs_full_values)
    n_model_values = list(n_model_values)

    adaptive_eval_mean_grid = np.full(
        (len(n_obs_full_values), len(n_model_values)),
        np.nan,
        dtype=np.float64,
    )
    adaptive_theta_mean_grid = np.full(
        (len(n_obs_full_values), len(n_model_values), 4),
        np.nan,
        dtype=np.float64,
    )
    baseline_eval_mean_grid = np.full_like(adaptive_eval_mean_grid, np.nan)
    natural_eval_mean_grid = np.full_like(adaptive_eval_mean_grid, np.nan)
    output_paths = np.empty((len(n_obs_full_values), len(n_model_values)), dtype=object)

    for obs_idx, n_obs_full in enumerate(n_obs_full_values):
        for model_idx, n_model in enumerate(n_model_values):
            cell_kwargs = dict(kwargs)
            cell_kwargs["n_obs_full"] = n_obs_full
            if tie_target_batch_to_n_obs:
                cell_kwargs["target_batch_size"] = n_obs_full

            cell_output_dir = output_dir / (
                f"{file_prefix}_m_{_format_value_for_filename(n_obs_full)}"
                f"_n_{_format_value_for_filename(n_model)}"
            )
            cell_output_dir.mkdir(parents=True, exist_ok=True)
            summary = run_for_n_model(
                n_model=n_model,
                seeds=seeds,
                output_dir=cell_output_dir,
                **cell_kwargs,
            )

            adaptive_eval_mean_grid[obs_idx, model_idx] = summary["adaptive_eval_mean"]
            adaptive_theta_mean_grid[obs_idx, model_idx] = np.asarray(
                summary["adaptive_theta_mean"],
                dtype=np.float64,
            )
            if "baseline_eval_mean" in summary:
                baseline_eval_mean_grid[obs_idx, model_idx] = summary["baseline_eval_mean"]
            if "natural_eval_mean" in summary:
                natural_eval_mean_grid[obs_idx, model_idx] = summary["natural_eval_mean"]
            output_paths[obs_idx, model_idx] = summary["output_path"]

    aggregated = {
        "grid_name": np.asarray("observation_model_grid"),
        "n_obs_full_values": np.asarray(n_obs_full_values, dtype=np.int32),
        "n_model_values": np.asarray(n_model_values, dtype=np.int32),
        "adaptive_eval_mean_grid": adaptive_eval_mean_grid,
        "adaptive_theta_mean_grid": adaptive_theta_mean_grid,
        "output_paths": np.asarray(output_paths, dtype=str),
        "seeds": np.asarray(list(seeds), dtype=np.int32),
        "tie_target_batch_to_n_obs": np.asarray(tie_target_batch_to_n_obs, dtype=np.bool_),
    }

    if np.isfinite(baseline_eval_mean_grid).any():
        aggregated["baseline_eval_mean_grid"] = baseline_eval_mean_grid
    if np.isfinite(natural_eval_mean_grid).any():
        aggregated["natural_eval_mean_grid"] = natural_eval_mean_grid

    summary_path = output_dir / f"{file_prefix}_summary.npz"
    save_results(aggregated, summary_path)
    aggregated["summary_path"] = str(summary_path)
    return aggregated


def run_lengthscale_regularization_grid_for_n_model(
    n_model,
    seeds,
    output_dir,
    lengthscale_param,
    lengthscale_values,
    lambda_scales,
    secondary_lengthscale_param=None,
    secondary_lengthscale_values=None,
    file_prefix="g_and_k_lengthscale_regularization_grid",
    **kwargs,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_kwargs = dict(kwargs)
    run_kwargs.setdefault("run_baseline", False)
    run_kwargs.setdefault("run_natural", False)

    lengthscale_values = list(lengthscale_values)
    lambda_scales = list(lambda_scales)

    if secondary_lengthscale_param is None:
        eval_mean_grid = np.full((len(lambda_scales), len(lengthscale_values)), np.nan, dtype=np.float64)
        theta_mean_grid = np.full((len(lambda_scales), len(lengthscale_values), 4), np.nan, dtype=np.float64)
        output_paths = np.empty((len(lambda_scales), len(lengthscale_values)), dtype=object)

        for lambda_idx, lambda_scale in enumerate(lambda_scales):
            for ell_idx, lengthscale_value in enumerate(lengthscale_values):
                cell_kwargs = dict(run_kwargs)
                cell_kwargs["lambda_scale"] = lambda_scale
                cell_kwargs[lengthscale_param] = lengthscale_value
                sweep_output_dir = output_dir / (
                    f"{file_prefix}_{lengthscale_param}_{_format_value_for_filename(lengthscale_value)}"
                    f"_lambda_{_format_value_for_filename(lambda_scale)}"
                )
                sweep_output_dir.mkdir(parents=True, exist_ok=True)
                summary = run_for_n_model(
                    n_model=n_model,
                    seeds=seeds,
                    output_dir=sweep_output_dir,
                    **cell_kwargs,
                )
                eval_mean_grid[lambda_idx, ell_idx] = summary["adaptive_eval_mean"]
                theta_mean_grid[lambda_idx, ell_idx] = np.asarray(summary["adaptive_theta_mean"], dtype=np.float64)
                output_paths[lambda_idx, ell_idx] = summary["output_path"]

        aggregated = {
            "grid_name": np.asarray("lengthscale_regularization_grid"),
            "lengthscale_param": np.asarray(lengthscale_param),
            "lengthscale_values": np.asarray(lengthscale_values, dtype=np.float64),
            "lambda_scales": np.asarray(lambda_scales, dtype=np.float64),
            "adaptive_eval_mean_grid": eval_mean_grid,
            "adaptive_theta_mean_grid": theta_mean_grid,
            "output_paths": np.asarray(output_paths, dtype=str),
            "n_model": np.asarray(n_model, dtype=np.int32),
            "seeds": np.asarray(list(seeds), dtype=np.int32),
        }
        summary_path = output_dir / f"{file_prefix}_{lengthscale_param}_summary.npz"
    else:
        secondary_lengthscale_values = list(secondary_lengthscale_values)
        eval_mean_grid = np.full(
            (len(lambda_scales), len(secondary_lengthscale_values), len(lengthscale_values)),
            np.nan,
            dtype=np.float64,
        )
        theta_mean_grid = np.full(
            (len(lambda_scales), len(secondary_lengthscale_values), len(lengthscale_values), 4),
            np.nan,
            dtype=np.float64,
        )
        output_paths = np.empty(
            (len(lambda_scales), len(secondary_lengthscale_values), len(lengthscale_values)),
            dtype=object,
        )

        for lambda_idx, lambda_scale in enumerate(lambda_scales):
            for secondary_idx, secondary_lengthscale_value in enumerate(secondary_lengthscale_values):
                for ell_idx, lengthscale_value in enumerate(lengthscale_values):
                    cell_kwargs = dict(run_kwargs)
                    cell_kwargs["lambda_scale"] = lambda_scale
                    cell_kwargs[lengthscale_param] = lengthscale_value
                    cell_kwargs[secondary_lengthscale_param] = secondary_lengthscale_value
                    sweep_output_dir = output_dir / (
                        f"{file_prefix}_{lengthscale_param}_{_format_value_for_filename(lengthscale_value)}"
                        f"_{secondary_lengthscale_param}_{_format_value_for_filename(secondary_lengthscale_value)}"
                        f"_lambda_{_format_value_for_filename(lambda_scale)}"
                    )
                    sweep_output_dir.mkdir(parents=True, exist_ok=True)
                    summary = run_for_n_model(
                        n_model=n_model,
                        seeds=seeds,
                        output_dir=sweep_output_dir,
                        **cell_kwargs,
                    )
                    eval_mean_grid[lambda_idx, secondary_idx, ell_idx] = summary["adaptive_eval_mean"]
                    theta_mean_grid[lambda_idx, secondary_idx, ell_idx] = np.asarray(
                        summary["adaptive_theta_mean"],
                        dtype=np.float64,
                    )
                    output_paths[lambda_idx, secondary_idx, ell_idx] = summary["output_path"]

        aggregated = {
            "grid_name": np.asarray("lengthscale_pair_regularization_grid"),
            "lengthscale_param": np.asarray(lengthscale_param),
            "lengthscale_values": np.asarray(lengthscale_values, dtype=np.float64),
            "secondary_lengthscale_param": np.asarray(secondary_lengthscale_param),
            "secondary_lengthscale_values": np.asarray(secondary_lengthscale_values, dtype=np.float64),
            "lambda_scales": np.asarray(lambda_scales, dtype=np.float64),
            "adaptive_eval_mean_grid": eval_mean_grid,
            "adaptive_theta_mean_grid": theta_mean_grid,
            "output_paths": np.asarray(output_paths, dtype=str),
            "n_model": np.asarray(n_model, dtype=np.int32),
            "seeds": np.asarray(list(seeds), dtype=np.int32),
        }
        summary_path = output_dir / (
            f"{file_prefix}_{lengthscale_param}_{secondary_lengthscale_param}_summary.npz"
        )

    save_results(aggregated, summary_path)
    aggregated["summary_path"] = str(summary_path)
    return aggregated


# ============================================================
# 8. Main experiment configuration
# ============================================================
# Change theta0 and hyperparameters here when running this script directly.
if __name__ == "__main__":
    experiment_mode = "single_run"

    # single_run_theta0_by_seed = np.array(
    #     [
    #         [3.5, 2.0, 0.6, -0.8],
    #         [3.4, 1.9, 0.7, -0.75],
    #         [2.0, 2.1, 0.5, -0.85],
    #         [2.0, 2.0, 1.3, -0.6],
    #         [3.7, 1.8, 0.6, -0.9],
    #     ],
    #     dtype=np.float64,
    # )
    single_run_kwargs = dict(
        theta_true=np.array([3.0, 1.0, 1.0, -np.log(2.0)], dtype=np.float64),
        theta0_by_seed=np.tile(np.array([3.5, 2.0, 0.6, -0.8], dtype=np.float64), (10, 1)),
        n_obs_full=1000,
        target_batch_size=600,
        n_steps_sgd=3000,
        n_steps_pgd=3000,
        gamma_sgd=0.1,
        gamma_natural_sgd=0.1,
        gamma_pgd0=0.1,
        lambda_scale=1e-3,
        natural_damping=1e-3,
        n_eval_model=2000,
        ell_fixed=2.0,
        ell0=10.0,
        ell_min=2.0,
        decay=0.99,
        run_adaptive_sgd=True,
        run_fixed_pgd=True,
    )
    ablation_kwargs = dict(
        theta_true=np.array([3.0, 1.0, 1.0, -np.log(2.0)], dtype=np.float64),
        theta0_by_seed=np.array(
            [
                [3.5, 2.0, 0.6, -0.8],
                [3.4, 1.9, 0.7, -0.75],
                [2.0, 2.1, 0.5, -0.85],
                [2.0, 2.0, 1.3, -0.6],
                [3.7, 1.8, 0.6, -0.9],
            ],
            dtype=np.float64,
        ),
        n_obs_full=1000,
        target_batch_size=200,
        n_steps_sgd=1,
        n_steps_pgd=3000,
        gamma_sgd=0.1,
        gamma_natural_sgd=0.1,
        gamma_pgd0=0.1,
        lambda_scale=1e-3,
        natural_damping=1e-3,
        n_eval_model=2000,
        ell_fixed=2.0,
        ell0=10.0,
        ell_min=2.0,
        decay=0.99,
        run_adaptive_sgd=False,
        run_fixed_pgd=False,
    )
    lengthscale_sweep_param = "ell_min"
    lengthscale_sweep_values = [0.1, 0.3, 1.0, 3.0, 10.0]
    lengthscale_grid_param = "ell_min"
    lengthscale_grid_values = [0.1, 0.3, 1.0, 3.0, 10.0]
    secondary_lengthscale_grid_param = "ell0"
    secondary_lengthscale_grid_values = [2.0, 5.0, 10.0, 20.0]
    decay_sweep_values = [0.99]

    if experiment_mode == "single_run":
        result = run_for_n_model(
            n_model=600,
            seeds=range(10),
            output_dir="/Users/sophiakang/Documents/GitHub/MDF_AL/results/gnk",
            **single_run_kwargs,
        )

        print("\nSaved results")
        print(f"baseline_eval_mean={result['baseline_eval_mean']:.8f}")
        print(f"natural_eval_mean={result['natural_eval_mean']:.8f}")
        print(f"adaptive_eval_mean={result['adaptive_eval_mean']:.8f}")
        print(f"baseline_theta_mean={np.array(result['baseline_theta_mean'])}")
        print(f"natural_theta_mean={np.array(result['natural_theta_mean'])}")
        print(f"adaptive_theta_mean={np.array(result['adaptive_theta_mean'])}")
        print(f"file={result['output_path']}")

    elif experiment_mode == "step_size_ablation":
        result = run_step_size_ablation_for_n_model(
            n_model=600,
            seeds=range(5),
            output_dir="ablations/gk_gamma",
            gamma_values=[10],#step_size_ablation
            sweep_param="gamma_pgd0",
            **ablation_kwargs,
        )

        print("\nSaved step-size ablation")
        print(f"summary={result['summary_path']}")

    elif experiment_mode == "decay_ablation":
        result = run_decay_ablation_for_n_model(
            n_model=600,
            seeds=range(5),
            output_dir="ablations/gk_decay",
            decay_values=decay_sweep_values,
            sweep_param="decay",
            **ablation_kwargs,
        )

        print("\nSaved decay ablation")
        print(f"summary={result['summary_path']}")

    elif experiment_mode == "observation_model_grid":
        result = run_observation_model_grid(
            seeds=range(5),
            n_obs_full_values=[300, 600, 1000],
            n_model_values=[10, 50],
            output_dir="ablations/gk_mn_grid",
            tie_target_batch_to_n_obs=True,
            **ablation_kwargs,
        )

        print("\nSaved observation/model grid")
        print(f"summary={result['summary_path']}")

    elif experiment_mode == "lengthscale_ablation":
        result = run_lengthscale_ablation_for_n_model(
            n_model=600,
            seeds=range(5),
            output_dir="ablations/gk_lengthscale",
            sweep_param=lengthscale_sweep_param,
            sweep_values=lengthscale_sweep_values,
            **ablation_kwargs,
        )

        print("\nSaved lengthscale ablation")
        print(f"summary={result['summary_path']}")

    elif experiment_mode == "regularization_ablation":
        result = run_regularization_ablation_for_n_model(
            n_model=600,
            seeds=range(5),
            output_dir="ablations/gk_ridge",
            lambda_scales=[1e-3],
            **ablation_kwargs,
        )

        print("\nSaved regularization ablation")
        print(f"summary={result['summary_path']}")

    elif experiment_mode == "lengthscale_regularization_grid":
        result = run_lengthscale_regularization_grid_for_n_model(
            n_model=600,
            seeds=range(5),
            output_dir="ablations/gk_heatmap",
            lengthscale_param=lengthscale_grid_param,
            lengthscale_values=lengthscale_grid_values,
            lambda_scales=[1e-4, 1e-2, 1e0],
            secondary_lengthscale_param=secondary_lengthscale_grid_param,
            secondary_lengthscale_values=secondary_lengthscale_grid_values,
            **ablation_kwargs,
        )

        print("\nSaved lengthscale/regularization grid")
        print(f"summary={result['summary_path']}")

    else:
        raise ValueError(
            "experiment_mode must be one of "
            "'single_run', 'step_size_ablation', 'decay_ablation', 'observation_model_grid', "
            "'lengthscale_ablation', 'regularization_ablation', or "
            "'lengthscale_regularization_grid'."
        )

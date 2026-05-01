import numpy as np
from functools import partial
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import ndtri


# ============================================================
# 1. g-and-k quantile + manual Jacobian
# ============================================================
def gk_quantile(u, theta):
    """
    u: scalar in (0,1)
    theta: [a, b, c, k]
    """
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
    """
    Jacobian wrt theta = [a, b, c, k]
    returns shape (4,)
    """
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


# ============================================================
# 2. Sampling
# ============================================================
def sample_gk(key, theta, n):
    eps = 1e-6
    u = jax.random.uniform(key, shape=(n,), minval=eps, maxval=1.0 - eps)
    x = gk_quantile_vmap(u, theta)
    return u, x


# ============================================================
# 3. Kernel / MMD / witness gradient
# ============================================================
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
    """
    x: model samples, shape (n_model,)
    y: target minibatch/full target, shape (batch_size,)
    returns grad f(x_i), shape (n_model,)
    """
    diff_xx = x[:, None] - x[None, :]
    k_xx = jnp.exp(-0.5 * (diff_xx**2) / (ell**2))
    grad_emp = jnp.mean(k_xx * (-diff_xx) / (ell**2), axis=1)

    diff_yx = y[:, None] - x[None, :]
    k_yx = jnp.exp(-0.5 * (diff_yx**2) / (ell**2))
    grad_tar = jnp.mean(k_yx * (diff_yx) / (ell**2), axis=0)

    return grad_emp - grad_tar


# ============================================================
# 4. Lengthscale schedules
# ============================================================
def make_fixed_ell_schedule(n_steps, ell_fixed):
    return np.full(n_steps, ell_fixed, dtype=np.float64)


def make_adaptive_ell_schedule(n_steps, ell0, ell_min, decay):
    ts = np.arange(n_steps, dtype=np.float64)
    return np.maximum(ell_min, ell0 * (decay ** ts))


# ============================================================
# 5. Parameter reparameterization
# ============================================================
def theta_min():
    return jnp.array([2.0, 0.3, 0.5, -1.0], dtype=jnp.float64)


def theta_max():
    return jnp.array([4.0, 1.8, 2.0, 0.0], dtype=jnp.float64)


def theta_center():
    return 0.5 * (theta_min() + theta_max())


def theta_width():
    return 0.5 * (theta_max() - theta_min())


def make_theta0(a0, b0):
    return jnp.array([3.75, 2.0, 0.5, -1.0], dtype=jnp.float64)


def phi_to_theta(phi, theta_ref):
    del theta_ref
    theta = theta_center() + theta_width() * phi
    return theta.at[1].set(jnp.maximum(theta[1], 1e-6))


def theta_to_phi(theta):
    return (theta - theta_center()) / theta_width()


# ============================================================
# 6. One SGD step (baseline: fixed ell, no preconditioning)
# ============================================================
@partial(jax.jit, static_argnums=(3,))
def gd_step(theta, key_model, y_batch, n_model, ell_t, gamma_t):
    u_model, x_model = sample_gk(key_model, theta, n_model)

    grad_f = witness_gradient_empirical(x_model, y_batch, ell_t)
    J = gk_jac_theta_vmap(u_model, theta)

    grad_theta = jnp.mean(J * grad_f[:, None], axis=0)

    theta_new = theta - gamma_t * grad_theta
    theta_new = theta_new.at[1].set(jnp.maximum(theta_new[1], 1e-6))

    train_loss = mmd2_vstat_1d(x_model, y_batch, ell_t)
    return theta_new, train_loss


# ============================================================
# 7. One PGD step (adaptive method)
# ============================================================
@partial(jax.jit, static_argnums=(4, 5))
def pgd_step(theta, key_model_A, key_model_b, y_batch, n_model_A, n_model_b, ell_t, gamma_t, lambda_t):
    u_model_A, _ = sample_gk(key_model_A, theta, n_model_A)
    J_A = gk_jac_theta_vmap(u_model_A, theta)

    u_model_b, x_model_b = sample_gk(key_model_b, theta, n_model_b)
    grad_f = witness_gradient_empirical(x_model_b, y_batch, ell_t)
    J_b = gk_jac_theta_vmap(u_model_b, theta)

    A = (J_A.T @ J_A) / n_model_A + lambda_t * jnp.eye(theta.shape[0], dtype=theta.dtype)
    b = jnp.mean(J_b * grad_f[:, None], axis=0)

    delta = jnp.linalg.solve(A, b)
    theta_new = theta - gamma_t * delta
    theta_new = theta_new.at[1].set(jnp.maximum(theta_new[1], 1e-6))

    train_loss = mmd2_vstat_1d(x_model_b, y_batch, ell_t)
    return theta_new, train_loss


# ============================================================
# 8. One rescaled SGD/PGD step in phi-space
# ============================================================
@partial(jax.jit, static_argnums=(4,))
def gd_step_phi(phi, theta_ref, key_model, y_batch, n_model, ell_t, gamma_t):
    theta = phi_to_theta(phi, theta_ref)
    u_model, x_model = sample_gk(key_model, theta, n_model)

    grad_f = witness_gradient_empirical(x_model, y_batch, ell_t)
    J = gk_jac_theta_vmap(u_model, theta)
    grad_theta = jnp.mean(J * grad_f[:, None], axis=0)

    grad_phi = grad_theta * theta_width()
    phi_new = phi - gamma_t * grad_phi

    train_loss = mmd2_vstat_1d(x_model, y_batch, ell_t)
    return phi_new, train_loss


@partial(jax.jit, static_argnums=(5, 6))
def pgd_step_phi(phi, theta_ref, key_model_A, key_model_b, y_batch, n_model_A, n_model_b, ell_t, gamma_t, lambda_t):
    theta = phi_to_theta(phi, theta_ref)
    u_model_A, _ = sample_gk(key_model_A, theta, n_model_A)
    J_theta_A = gk_jac_theta_vmap(u_model_A, theta)
    J_phi_A = J_theta_A * theta_width()[None, :]

    u_model_b, x_model_b = sample_gk(key_model_b, theta, n_model_b)
    grad_f = witness_gradient_empirical(x_model_b, y_batch, ell_t)
    J_theta_b = gk_jac_theta_vmap(u_model_b, theta)
    J_phi_b = J_theta_b * theta_width()[None, :]

    A = (J_phi_A.T @ J_phi_A) / n_model_A + lambda_t * jnp.eye(phi.shape[0], dtype=phi.dtype)
    b = jnp.mean(J_phi_b * grad_f[:, None], axis=0)

    delta = jnp.linalg.solve(A, b)
    phi_new = phi - gamma_t * delta

    train_loss = mmd2_vstat_1d(x_model_b, y_batch, ell_t)
    return phi_new, train_loss


# ============================================================
# 9. Full evaluation loss
# ============================================================
@partial(jax.jit, static_argnums=(2,))
def eval_loss_full(theta, key_eval, n_eval_model, y_obs_full, ell_t):
    _, x_eval = sample_gk(key_eval, theta, n_eval_model)
    return mmd2_vstat_1d(x_eval, y_obs_full, ell_t)


# ============================================================
# 10. Shared initialization
# ============================================================
def make_target_and_init(seed, theta_true, n_obs_full, theta0=None):
    key = jax.random.PRNGKey(seed)
    key, key_obs = jax.random.split(key)

    _, y_obs_full = sample_gk(key_obs, theta_true, n_obs_full)

    if theta0 is None:
        theta0 = jnp.array([2.0, 2.0, 1.5, -0.3], dtype=jnp.float64)
    else:
        theta0 = jnp.array(theta0, dtype=jnp.float64)

    return y_obs_full, theta0


# ============================================================
# 11. Baseline SGD run
# ============================================================
def run_baseline_sgd(
    seed,
    theta0,
    y_obs_full,
    target_batch_size,
    n_model,
    n_steps_sgd,
    gamma_sgd,
    ell_fixed,
    n_eval_model=2000,
    print_every=20,
    history_every=10,
):
    key = jax.random.PRNGKey(seed + 1000)
    theta_ref = theta0
    phi = theta_to_phi(theta0)
    last_train_loss = None
    last_eval_loss = None
    history_steps = []
    train_loss_history = []
    eval_loss_history = []
    theta_history = []

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

        phi, train_loss = gd_step_phi(
            phi=phi,
            theta_ref=theta_ref,
            key_model=key_model,
            y_batch=y_batch,
            n_model=n_model,
            ell_t=ell_t,
            gamma_t=jnp.asarray(gamma_sgd, dtype=jnp.float64),
        )

        last_train_loss = train_loss

        if (t % history_every == 0) or (t == n_steps_sgd - 1):
            theta = phi_to_theta(phi, theta_ref)
            eval_loss = eval_loss_full(
                theta=theta,
                key_eval=key_eval,
                n_eval_model=n_eval_model,
                y_obs_full=y_obs_full,
                ell_t=ell_t,
            )
            last_eval_loss = eval_loss
            history_steps.append(t)
            train_loss_history.append(float(train_loss))
            eval_loss_history.append(float(eval_loss))
            theta_history.append(np.array(theta, dtype=np.float64))

        if (t % print_every == 0) or (t == n_steps_sgd - 1):
            if last_eval_loss is None:
                theta = phi_to_theta(phi, theta_ref)
                last_eval_loss = eval_loss_full(
                    theta=theta,
                    key_eval=key_eval,
                    n_eval_model=n_eval_model,
                    y_obs_full=y_obs_full,
                    ell_t=ell_t,
                )

            print(
                f"[SGD] step={t:4d} | ell={float(ell_t):.4f} | "
                f"gamma={float(gamma_sgd):.6f} | "
                f"train_loss={float(train_loss):.8f} | "
                f"eval_loss={float(last_eval_loss):.8f}"
            )

    theta = phi_to_theta(phi, theta_ref)
    return {
        "theta_final": np.array(theta, dtype=np.float64),
        "train_loss_final": float(last_train_loss),
        "eval_loss_final": float(last_eval_loss) if last_eval_loss is not None else None,
        "history_steps": np.array(history_steps, dtype=np.int32),
        "train_loss_history": np.array(train_loss_history, dtype=np.float64),
        "eval_loss_history": np.array(eval_loss_history, dtype=np.float64),
        "theta_history": np.array(theta_history, dtype=np.float64),
    }


# ============================================================
# 12. Adaptive PGD run with decreasing step size
# ============================================================
def run_adaptive_pgd(
    seed,
    theta0,
    y_obs_full,
    target_batch_size,
    n_model_A,
    n_model_b,
    n_steps_pgd,
    gamma_pgd0,
    lambda_scale,
    ell0,
    ell_min,
    decay,
    n_eval_model=2000,
    print_every=20,
    history_every=10,
):
    key = jax.random.PRNGKey(seed + 2000)
    theta_ref = theta0
    phi = theta_to_phi(theta0)
    last_train_loss = None
    last_eval_loss = None
    history_steps = []
    train_loss_history = []
    eval_loss_history = []
    theta_history = []

    ell_schedule = make_adaptive_ell_schedule(n_steps_pgd, ell0, ell_min, decay)

    for t in range(n_steps_pgd):
        key, key_batch, key_model_A, key_model_b, key_eval = jax.random.split(key, 5)

        idx = jax.random.randint(
            key_batch,
            shape=(target_batch_size,),
            minval=0,
            maxval=y_obs_full.shape[0],
        )
        y_batch = y_obs_full[idx]

        ell_t = jnp.asarray(ell_schedule[t], dtype=jnp.float64)
        gamma_t = jnp.asarray(gamma_pgd0 / (t + 1.0)**0.15, dtype=jnp.float64)
        lambda_t = jnp.asarray(
            lambda_scale
            * ((float(gamma_t) * (float(ell_min) ** -1.0) * (float(ell_t) ** -2.0)) ** (2.0 / 5.0)),
            dtype=jnp.float64,
        )

        phi, train_loss = pgd_step_phi(
            phi=phi,
            theta_ref=theta_ref,
            key_model_A=key_model_A,
            key_model_b=key_model_b,
            y_batch=y_batch,
            n_model_A=n_model_A,
            n_model_b=n_model_b,
            ell_t=ell_t,
            gamma_t=gamma_t,
            lambda_t=lambda_t,
        )

        last_train_loss = train_loss

        if (t % history_every == 0) or (t == n_steps_pgd - 1):
            theta = phi_to_theta(phi, theta_ref)
            eval_loss = eval_loss_full(
                theta=theta,
                key_eval=key_eval,
                n_eval_model=n_eval_model,
                y_obs_full=y_obs_full,
                ell_t=ell_t,
            )
            last_eval_loss = eval_loss
            history_steps.append(t)
            train_loss_history.append(float(train_loss))
            eval_loss_history.append(float(eval_loss))
            theta_history.append(np.array(theta, dtype=np.float64))

        if (t % print_every == 0) or (t == n_steps_pgd - 1):
            if last_eval_loss is None:
                theta = phi_to_theta(phi, theta_ref)
                last_eval_loss = eval_loss_full(
                    theta=theta,
                    key_eval=key_eval,
                    n_eval_model=n_eval_model,
                    y_obs_full=y_obs_full,
                    ell_t=ell_t,
                )

            print(
                f"[PGD] step={t:4d} | ell={float(ell_t):.4f} | "
                f"gamma={float(gamma_t):.6f} | lambda={float(lambda_t):.6f} | "
                f"train_loss={float(train_loss):.8f} | "
                f"eval_loss={float(last_eval_loss):.8f}"
            )

    theta = phi_to_theta(phi, theta_ref)
    return {
        "theta_final": np.array(theta, dtype=np.float64),
        "train_loss_final": float(last_train_loss),
        "eval_loss_final": float(last_eval_loss) if last_eval_loss is not None else None,
        "history_steps": np.array(history_steps, dtype=np.int32),
        "train_loss_history": np.array(train_loss_history, dtype=np.float64),
        "eval_loss_history": np.array(eval_loss_history, dtype=np.float64),
        "theta_history": np.array(theta_history, dtype=np.float64),
    }


# ============================================================
# 13. Compare baseline SGD vs adaptive PGD
# ============================================================
def run_baseline_and_adaptive(
    seed=0,
    theta_true=np.array([3.0, 1.0, 1.0, -np.log(2.0)], dtype=np.float64),
    theta0=None,
    n_obs_full=500,
    target_batch_size=200,
    n_model=300,
    n_model_pgd_A=500,
    n_model_pgd_b=100,
    n_steps_sgd=5000,
    n_steps_pgd=2000,
    gamma_sgd=0.01,
    gamma_pgd0=0.49,
    lambda_scale=1e-3,
    n_eval_model=2000,
    ell_fixed=2.0,
    ell0=20.0,
    ell_min=2.0,
    decay=0.997,
    history_every=10,
):
    theta_true = jnp.array(theta_true, dtype=jnp.float64)

    y_obs_full, theta0 = make_target_and_init(seed, theta_true, n_obs_full, theta0=theta0)

    print("\n=== Plain SGD baseline (fixed lengthscale, no preconditioning) ===")
    baseline_res = run_baseline_sgd(
        seed=seed,
        theta0=theta0,
        y_obs_full=y_obs_full,
        target_batch_size=target_batch_size,
        n_model=n_model,
        n_steps_sgd=n_steps_sgd,
        gamma_sgd=gamma_sgd,
        ell_fixed=ell_fixed,
        n_eval_model=n_eval_model,
        print_every=20,
        history_every=history_every,
    )

    print("\n=== Adaptive PGD run ===")
    adaptive_res = run_adaptive_pgd(
        seed=seed,
        theta0=theta0,
        y_obs_full=y_obs_full,
        target_batch_size=target_batch_size,
        n_model_A=n_model_pgd_A,
        n_model_b=n_model_pgd_b,
        n_steps_pgd=n_steps_pgd,
        gamma_pgd0=gamma_pgd0,
        lambda_scale=lambda_scale,
        ell0=ell0,
        ell_min=ell_min,
        decay=decay,
        n_eval_model=n_eval_model,
        print_every=20,
        history_every=history_every,
    )

    return {
        "baseline_theta_final": baseline_res["theta_final"],
        "baseline_train_loss_final": baseline_res["train_loss_final"],
        "baseline_eval_loss_final": baseline_res["eval_loss_final"],
        "baseline_history_steps": baseline_res["history_steps"],
        "baseline_train_loss_history": baseline_res["train_loss_history"],
        "baseline_eval_loss_history": baseline_res["eval_loss_history"],
        "baseline_theta_history": baseline_res["theta_history"],
        "adaptive_theta_final": adaptive_res["theta_final"],
        "adaptive_train_loss_final": adaptive_res["train_loss_final"],
        "adaptive_eval_loss_final": adaptive_res["eval_loss_final"],
        "adaptive_history_steps": adaptive_res["history_steps"],
        "adaptive_train_loss_history": adaptive_res["adaptive_train_loss_history"] if "adaptive_train_loss_history" in adaptive_res else adaptive_res["train_loss_history"],
        "adaptive_eval_loss_history": adaptive_res["adaptive_eval_loss_history"] if "adaptive_eval_loss_history" in adaptive_res else adaptive_res["eval_loss_history"],
        "adaptive_theta_history": adaptive_res["adaptive_theta_history"] if "adaptive_theta_history" in adaptive_res else adaptive_res["theta_history"],
    }


# ============================================================
# 14. Multi-seed / multi-n_model runner
# ============================================================
def save_results(results, output_path):
    np.savez_compressed(output_path, **results)


def _format_theta_for_filename(theta):
    parts = []
    for value in np.asarray(theta, dtype=np.float64):
        token = f"{value:.3f}".replace("-", "m").replace(".", "p")
        parts.append(token)
    return "_".join(parts)


def run_grid_over_n_model(
    seeds,
    n_models,
    output_dir,
    theta_true=np.array([3.0, 1.0, 1.0, -np.log(2.0)], dtype=np.float64),
    theta0=None,
    n_obs_full=500,
    target_batch_size=200,
    n_model_pgd_A=500,
    n_model_pgd_b=100,
    n_steps_sgd=5000,
    n_steps_pgd=2000,
    gamma_sgd=0.01,
    gamma_pgd0=0.49,
    lambda_scale=1e-3,
    n_eval_model=2000,
    ell_fixed=2.0,
    ell0=20.0,
    ell_min=2.0,
    decay=0.997,
    history_every=10,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    theta0_value = np.array([2.0, 2.0, 1.5, -0.3], dtype=np.float64) if theta0 is None else np.array(theta0, dtype=np.float64)
    theta0_tag = _format_theta_for_filename(theta0_value)

    for n_model in n_models:
        baseline_train = []
        baseline_eval = []
        adaptive_train = []
        adaptive_eval = []
        baseline_thetas = []
        adaptive_thetas = []
        baseline_history_steps = None
        adaptive_history_steps = None
        baseline_train_histories = []
        baseline_eval_histories = []
        adaptive_train_histories = []
        adaptive_eval_histories = []
        baseline_theta_histories = []
        adaptive_theta_histories = []

        for seed in seeds:
            res = run_baseline_and_adaptive(
                seed=seed,
                theta_true=theta_true,
                theta0=theta0_value,
                n_obs_full=n_obs_full,
                target_batch_size=target_batch_size,
                n_model=n_model,
                n_model_pgd_A=n_model_pgd_A,
                n_model_pgd_b=n_model_pgd_b,
                n_steps_sgd=n_steps_sgd,
                n_steps_pgd=n_steps_pgd,
                gamma_sgd=gamma_sgd,
                gamma_pgd0=gamma_pgd0,
                lambda_scale=lambda_scale,
                n_eval_model=n_eval_model,
                ell_fixed=ell_fixed,
                ell0=ell0,
                ell_min=ell_min,
                decay=decay,
                history_every=history_every,
            )

            baseline_train.append(res["baseline_train_loss_final"])
            baseline_eval.append(res["baseline_eval_loss_final"])
            adaptive_train.append(res["adaptive_train_loss_final"])
            adaptive_eval.append(res["adaptive_eval_loss_final"])
            baseline_thetas.append(res["baseline_theta_final"])
            adaptive_thetas.append(res["adaptive_theta_final"])
            if baseline_history_steps is None:
                baseline_history_steps = np.array(res["baseline_history_steps"], dtype=np.int32)
            if adaptive_history_steps is None:
                adaptive_history_steps = np.array(res["adaptive_history_steps"], dtype=np.int32)
            baseline_train_histories.append(res["baseline_train_loss_history"])
            baseline_eval_histories.append(res["baseline_eval_loss_history"])
            adaptive_train_histories.append(res["adaptive_train_loss_history"])
            adaptive_eval_histories.append(res["adaptive_eval_loss_history"])
            baseline_theta_histories.append(res["baseline_theta_history"])
            adaptive_theta_histories.append(res["adaptive_theta_history"])

        baseline_train = np.array(baseline_train, dtype=np.float64)
        baseline_eval = np.array(baseline_eval, dtype=np.float64)
        adaptive_train = np.array(adaptive_train, dtype=np.float64)
        adaptive_eval = np.array(adaptive_eval, dtype=np.float64)
        baseline_thetas = np.array(baseline_thetas, dtype=np.float64)
        adaptive_thetas = np.array(adaptive_thetas, dtype=np.float64)
        baseline_train_histories = np.array(baseline_train_histories, dtype=np.float64)
        baseline_eval_histories = np.array(baseline_eval_histories, dtype=np.float64)
        adaptive_train_histories = np.array(adaptive_train_histories, dtype=np.float64)
        adaptive_eval_histories = np.array(adaptive_eval_histories, dtype=np.float64)
        baseline_theta_histories = np.array(baseline_theta_histories, dtype=np.float64)
        adaptive_theta_histories = np.array(adaptive_theta_histories, dtype=np.float64)

        results = {
            "seeds": np.array(seeds, dtype=np.int32),
            "n_model": np.array(n_model, dtype=np.int32),
            "n_model_pgd_A": np.array(n_model_pgd_A, dtype=np.int32),
            "n_model_pgd_b": np.array(n_model_pgd_b, dtype=np.int32),
            "theta_true": np.array(theta_true, dtype=np.float64),
            "theta0": np.array(theta0_value, dtype=np.float64),
            "n_obs_full": np.array(n_obs_full, dtype=np.int32),
            "target_batch_size": np.array(target_batch_size, dtype=np.int32),
            "n_steps_sgd": np.array(n_steps_sgd, dtype=np.int32),
            "n_steps_pgd": np.array(n_steps_pgd, dtype=np.int32),
            "gamma_sgd": np.array(gamma_sgd, dtype=np.float64),
            "gamma_pgd0": np.array(gamma_pgd0, dtype=np.float64),
            "lambda_scale": np.array(lambda_scale, dtype=np.float64),
            "n_eval_model": np.array(n_eval_model, dtype=np.int32),
            "ell_fixed": np.array(ell_fixed, dtype=np.float64),
            "ell0": np.array(ell0, dtype=np.float64),
            "ell_min": np.array(ell_min, dtype=np.float64),
            "decay": np.array(decay, dtype=np.float64),
            "baseline_train_losses": baseline_train,
            "baseline_eval_losses": baseline_eval,
            "adaptive_train_losses": adaptive_train,
            "adaptive_eval_losses": adaptive_eval,
            "baseline_thetas": baseline_thetas,
            "adaptive_thetas": adaptive_thetas,
            "baseline_theta_mean": np.mean(baseline_thetas, axis=0),
            "baseline_theta_std": np.std(baseline_thetas, axis=0),
            "adaptive_theta_mean": np.mean(adaptive_thetas, axis=0),
            "adaptive_theta_std": np.std(adaptive_thetas, axis=0),
            "baseline_train_mean": np.mean(baseline_train),
            "baseline_train_std": np.std(baseline_train),
            "baseline_eval_mean": np.mean(baseline_eval),
            "baseline_eval_std": np.std(baseline_eval),
            "adaptive_train_mean": np.mean(adaptive_train),
            "adaptive_train_std": np.std(adaptive_train),
            "adaptive_eval_mean": np.mean(adaptive_eval),
            "adaptive_eval_std": np.std(adaptive_eval),
            "history_every": np.array(history_every, dtype=np.int32),
            "baseline_history_steps": baseline_history_steps,
            "adaptive_history_steps": adaptive_history_steps,
            "baseline_train_histories": baseline_train_histories,
            "baseline_eval_histories": baseline_eval_histories,
            "adaptive_train_histories": adaptive_train_histories,
            "adaptive_eval_histories": adaptive_eval_histories,
            "baseline_theta_histories": baseline_theta_histories,
            "adaptive_theta_histories": adaptive_theta_histories,
            "baseline_train_history_mean": np.mean(baseline_train_histories, axis=0),
            "baseline_eval_history_mean": np.mean(baseline_eval_histories, axis=0),
            "adaptive_train_history_mean": np.mean(adaptive_train_histories, axis=0),
            "adaptive_eval_history_mean": np.mean(adaptive_eval_histories, axis=0),
            "baseline_theta_history_mean": np.mean(baseline_theta_histories, axis=0),
            "adaptive_theta_history_mean": np.mean(adaptive_theta_histories, axis=0),
        }

        output_path = output_dir / f"g_n_k_fixed{n_model}_theta0_{theta0_tag}.npz"
        save_results(results, output_path)
        summary[int(n_model)] = {
            "output_path": str(output_path),
            "baseline_eval_mean": float(results["baseline_eval_mean"]),
            "adaptive_eval_mean": float(results["adaptive_eval_mean"]),
            "baseline_theta_mean": results["baseline_theta_mean"].tolist(),
            "adaptive_theta_mean": results["adaptive_theta_mean"].tolist(),
        }

    return summary


def run_for_n_model(
    n_model,
    seeds,
    output_dir,
    theta_true=np.array([3.0, 1.0, 1.0, -np.log(2.0)], dtype=np.float64),
    theta0=None,
    n_obs_full=500,
    target_batch_size=200,
    n_model_pgd_A=500,
    n_model_pgd_b=100,
    n_steps_sgd=5000,
    n_steps_pgd=2000,
    gamma_sgd=0.01,
    gamma_pgd0=0.49,
    lambda_scale=1e-3,
    n_eval_model=2000,
    ell_fixed=2.0,
    ell0=20.0,
    ell_min=2.0,
    decay=0.997,
    history_every=10,
):
    summary = run_grid_over_n_model(
        seeds=seeds,
        n_models=[n_model],
        output_dir=output_dir,
        theta0=theta0,
        n_model_pgd_A=n_model_pgd_A,
        n_model_pgd_b=n_model_pgd_b,
        theta_true=theta_true,
        n_obs_full=n_obs_full,
        target_batch_size=target_batch_size,
        n_steps_sgd=n_steps_sgd,
        n_steps_pgd=n_steps_pgd,
        gamma_sgd=gamma_sgd,
        gamma_pgd0=gamma_pgd0,
        lambda_scale=lambda_scale,
        n_eval_model=n_eval_model,
        ell_fixed=ell_fixed,
        ell0=ell0,
        ell_min=ell_min,
        decay=decay,
        history_every=history_every,
    )
    return summary[int(n_model)]


# ============================================================
# 15. Main
# ============================================================
if __name__ == "__main__":
    result = run_for_n_model(
        n_model=200,
        seeds=[0],
        output_dir="/Users/sophiakang/Documents/GitHub/MDF_AL",
        theta_true=np.array([3.0, 1.0, 1.0, -np.log(2.0)], dtype=np.float64),
        theta0=np.array([2.5, 1.5, 2.0, -0.25], dtype=np.float64),
        n_obs_full=1000,
        target_batch_size=200,
        n_model_pgd_A=150,
        n_model_pgd_b=50,
        n_steps_sgd=1000,
        n_steps_pgd=20000,
        gamma_sgd=0.5,   # step size for SGD
        gamma_pgd0=0.3, # initial step size for PGD; actual step is gamma_pgd0 / sqrt(t+1)
        lambda_scale=1e-3,
        n_eval_model=2000,
        ell_fixed=2.0,
        ell0=10.0,
        ell_min=2.0,
        decay=0.9985,
    )

    print("\nSaved results")
    print(f"baseline_eval_mean={result['baseline_eval_mean']:.8f}")
    print(f"adaptive_eval_mean={result['adaptive_eval_mean']:.8f}")
    print(f"file={result['output_path']}")
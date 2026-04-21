import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax


# ============================================================
# 1. Target: 8 isotropic Gaussians on a circle
# ============================================================
k = 8
d = 2
radius = 2.0
std = 0.2
angles = 2.0 * np.pi * np.arange(k, dtype=np.float64) / k

means = np.stack(
    [
        radius * np.cos(angles),
        radius * np.sin(angles),
    ],
    axis=1,
).astype(np.float64)

covs = np.array(
    [(std**2) * np.eye(d, dtype=np.float64) for _ in range(k)],
    dtype=np.float64,
)

weights = np.ones(k, dtype=np.float64) / k


# ============================================================
# 2. Kernel, objective, witness gradient
# ============================================================
def sq_dists(x, y):
    x2 = jnp.sum(x**2, axis=1, keepdims=True)
    y2 = jnp.sum(y**2, axis=1, keepdims=True).T
    return x2 + y2 - 2.0 * (x @ y.T)


def gaussian_kernel(x, y, ell):
    sq = sq_dists(x, y)
    return jnp.exp(-0.5 * sq / (ell**2))


def gaussian_mean_embedding(x, means, covs, weights, ell):
    dim = x.shape[1]
    eye = jnp.eye(dim, dtype=x.dtype)

    A = covs + (ell**2) * eye[None, :, :]
    A_inv = jnp.linalg.inv(A)

    diff = x[None, :, :] - means[:, None, :]
    solved = jnp.einsum("kde,kne->knd", A_inv, diff)
    mahal = jnp.einsum("knd,knd->kn", diff, solved)

    _, logdet_A = jnp.linalg.slogdet(A)
    log_pref = dim * jnp.log(ell) - 0.5 * logdet_A
    vals = jnp.exp(log_pref[:, None] - 0.5 * mahal)

    weighted_vals = weights[:, None] * vals
    kme_vals = jnp.sum(weighted_vals, axis=0)
    grad_vals = -jnp.sum(weighted_vals[:, :, None] * solved, axis=0)
    return kme_vals, grad_vals


def gaussian_mixture_kernel_expectation(means, covs, weights, ell):
    dim = means.shape[1]
    eye = jnp.eye(dim, dtype=means.dtype)

    pair_covs = covs[:, None, :, :] + covs[None, :, :, :]
    A = pair_covs + (ell**2) * eye[None, None, :, :]
    A_inv = jnp.linalg.inv(A)

    deltas = means[:, None, :] - means[None, :, :]
    solved = jnp.einsum("abde,abe->abd", A_inv, deltas)
    mahal = jnp.einsum("abd,abd->ab", deltas, solved)

    _, logdet_A = jnp.linalg.slogdet(A)
    log_pref = dim * jnp.log(ell) - 0.5 * logdet_A
    vals = jnp.exp(log_pref - 0.5 * mahal)

    pair_weights = weights[:, None] * weights[None, :]
    return jnp.sum(pair_weights * vals)


def F(x, means, covs, weights, ell):
    n = x.shape[0]
    kxx = gaussian_kernel(x, x, ell)

    term_xx = jnp.sum(kxx) / (n * n)
    term_yy = gaussian_mixture_kernel_expectation(means, covs, weights, ell)
    kme_vals, _ = gaussian_mean_embedding(x, means, covs, weights, ell)
    term_xy = 2.0 * jnp.mean(kme_vals)

    return 0.5 * (term_xx + term_yy - term_xy)


def F_with_precomputed_term_yy(x, means, covs, weights, ell, term_yy):
    n = x.shape[0]
    kxx = gaussian_kernel(x, x, ell)
    term_xx = jnp.sum(kxx) / (n * n)
    kme_vals, _ = gaussian_mean_embedding(x, means, covs, weights, ell)
    term_xy = 2.0 * jnp.mean(kme_vals)
    return 0.5 * (term_xx + term_yy - term_xy)


def witness_gradient(x, means, covs, weights, ell):
    diff_xx = x[:, None, :] - x[None, :, :]
    sq_xx = jnp.sum(diff_xx**2, axis=2)
    k_xx = jnp.exp(-0.5 * sq_xx / (ell**2))

    grad_emp = jnp.mean(k_xx[:, :, None] * (-diff_xx) / (ell**2), axis=1)
    _, grad_tar = gaussian_mean_embedding(x, means, covs, weights, ell)
    return grad_emp - grad_tar


# ============================================================
# 3. Lengthscale schedules
# ============================================================
def make_ell_schedule(n_steps, mode, ell0, ell_min, decay):
    ts = jnp.arange(n_steps, dtype=jnp.float64)

    if mode == "fixed":
        return jnp.full((n_steps,), ell0, dtype=jnp.float64)
    if mode == "adaptive":
        return jnp.maximum(ell_min, ell0 * (decay ** ts))
    raise ValueError("mode must be 'fixed' or 'adaptive'")


# ============================================================
# 4. Gradient flow
# ============================================================
@jax.jit
def mmd_gf_one_step(x, means, covs, weights, ell, step_size):
    grad = witness_gradient(x, means, covs, weights, ell)
    x_new = x - step_size * grad
    return x_new


def run_flow_fixed(x0, means, covs, weights, ell_schedule, step_size, eval_ell):
    def one_step(x, ell):
        x_new = mmd_gf_one_step(
            x=x,
            means=means,
            covs=covs,
            weights=weights,
            ell=ell,
            step_size=jnp.asarray(step_size, dtype=ell.dtype),
        )
        return x_new, None

    x_final, _ = lax.scan(one_step, x0, ell_schedule)
    eval_ell = jnp.asarray(eval_ell, dtype=x_final.dtype)
    term_yy = gaussian_mixture_kernel_expectation(means, covs, weights, eval_ell)
    f_final = F_with_precomputed_term_yy(x_final, means, covs, weights, eval_ell, term_yy)
    return np.array(x_final), float(f_final)


def run_flow_adaptive(x0, means, covs, weights, ell_schedule, eval_ell):
    def one_step(x, ell):
        x_new = mmd_gf_one_step(
            x=x,
            means=means,
            covs=covs,
            weights=weights,
            ell=ell,
            step_size=(ell**2) / 12.0,
        )
        return x_new, None

    x_final, _ = lax.scan(one_step, x0, ell_schedule)
    eval_ell = jnp.asarray(eval_ell, dtype=x_final.dtype)
    term_yy = gaussian_mixture_kernel_expectation(means, covs, weights, eval_ell)
    f_final = F_with_precomputed_term_yy(x_final, means, covs, weights, eval_ell, term_yy)
    return np.array(x_final), float(f_final)


def make_lhs_rhs_checkpoint_steps(n_steps):
    steps = np.arange(500, n_steps + 1, 500, dtype=np.int64)

    if steps.size == 0 or steps[-1] != n_steps:
        steps = np.append(steps, np.int64(n_steps))

    return steps


@jax.jit
def f_value(x, means, covs, weights, eval_ell, term_yy):
    return F_with_precomputed_term_yy(x, means, covs, weights, eval_ell, term_yy)


def run_flow_fixed_with_history(
    x0,
    means,
    covs,
    weights,
    ell_schedule,
    step_size,
    eval_ell,
    checkpoint_steps,
    stop_rel_tol=None,
    stop_patience=3,
    print_stop=False,
):
    def run_segment(x, ell_segment):
        def one_step(x_inner, ell):
            x_new = mmd_gf_one_step(
                x=x_inner,
                means=means,
                covs=covs,
                weights=weights,
                ell=ell,
                step_size=jnp.asarray(step_size, dtype=ell.dtype),
            )
            return x_new, None

        x_final, _ = lax.scan(one_step, x, ell_segment)
        return x_final

    n_steps = int(ell_schedule.shape[0])
    checkpoint_steps = np.asarray(checkpoint_steps, dtype=np.int64)
    checkpoint_steps = checkpoint_steps[
        (checkpoint_steps > 0) & (checkpoint_steps <= n_steps)
    ]
    checkpoint_steps = np.unique(checkpoint_steps)

    eval_ell = jnp.asarray(eval_ell, dtype=x0.dtype)
    term_yy = gaussian_mixture_kernel_expectation(means, covs, weights, eval_ell)

    f_values = []
    x = x0
    prev_step = 0
    prev_f = None
    small_change_count = 0
    stopped_early = False

    for step in checkpoint_steps:
        x = run_segment(x, ell_schedule[prev_step:step])
        f_values.append(float(f_value(x, means, covs, weights, eval_ell, term_yy)))
        f_current = f_values[-1]
        rel_change = None
        if prev_f is not None:
            rel_change = abs(prev_f - f_current) / max(abs(prev_f), 1e-300)
        prev_step = int(step)
        if stop_rel_tol is not None and rel_change is not None:
            if rel_change < stop_rel_tol:
                small_change_count += 1
            else:
                small_change_count = 0

            if small_change_count >= stop_patience:
                stopped_early = True
                if print_stop:
                    print(
                        f"Stopping fixed at step={int(step):d}: "
                        f"relative_change={rel_change:.6e} "
                        f"< tol={stop_rel_tol:.6e} "
                        f"for {stop_patience:d} checkpoints"
                    )
                break

        prev_f = f_current

    if prev_step < n_steps and not stopped_early:
        x = run_segment(x, ell_schedule[prev_step:n_steps])

    f_final = F_with_precomputed_term_yy(x, means, covs, weights, eval_ell, term_yy)
    return np.array(x), float(f_final), np.asarray(f_values, dtype=np.float64)


@jax.jit
def lhs_rhs_values(x, means, covs, weights, ell_t, ell_inf, term_yy_inf):
    grad = witness_gradient(x, means, covs, weights, ell_inf)
    scale = ((ell_t**2) / (ell_inf**2)) ** (0.5 * x.shape[1])
    lhs = scale * jnp.mean(jnp.sum(grad**2, axis=1))
    f_val = F_with_precomputed_term_yy(x, means, covs, weights, ell_inf, term_yy_inf)
    rhs = 2.0 * f_val
    return lhs, rhs


def run_flow_adaptive_with_lhs_rhs(
    x0,
    means,
    covs,
    weights,
    ell_schedule,
    ell_inf,
    eval_ell,
    checkpoint_steps,
    print_lhs_rhs=False,
    fixed_history_for_print=None,
    print_mean_history=False,
    stop_rel_tol=None,
    stop_patience=3,
):
    def run_segment(x, ell_segment):
        def one_step(x_inner, ell):
            x_new = mmd_gf_one_step(
                x=x_inner,
                means=means,
                covs=covs,
                weights=weights,
                ell=ell,
                step_size=(ell**2) / 12.0,
            )
            return x_new, None

        x_final, _ = lax.scan(one_step, x, ell_segment)
        return x_final

    n_steps = int(ell_schedule.shape[0])
    checkpoint_steps = np.asarray(checkpoint_steps, dtype=np.int64)
    checkpoint_steps = checkpoint_steps[
        (checkpoint_steps > 0) & (checkpoint_steps <= n_steps)
    ]
    checkpoint_steps = np.unique(checkpoint_steps)

    ell_inf = jnp.asarray(ell_inf, dtype=x0.dtype)
    eval_ell = jnp.asarray(eval_ell, dtype=x0.dtype)
    term_yy_inf = gaussian_mixture_kernel_expectation(means, covs, weights, ell_inf)
    term_yy_eval = gaussian_mixture_kernel_expectation(means, covs, weights, eval_ell)

    lhs_values = []
    rhs_values = []
    f_values = []
    x = x0
    prev_step = 0
    prev_f = None
    small_change_count = 0
    stopped_early = False

    for step in checkpoint_steps:
        x = run_segment(x, ell_schedule[prev_step:step])
        grad_ell = ell_schedule[step - 1]
        f_values.append(float(f_value(x, means, covs, weights, eval_ell, term_yy_eval)))
        f_current = f_values[-1]
        rel_change = None
        if prev_f is not None:
            rel_change = abs(prev_f - f_current) / max(abs(prev_f), 1e-300)

        lhs, rhs = lhs_rhs_values(
            x,
            means,
            covs,
            weights,
            grad_ell,
            ell_inf,
            term_yy_inf,
        )
        lhs_values.append(float(lhs))
        rhs_values.append(float(rhs))
        if print_lhs_rhs:
            lhs_float = lhs_values[-1]
            rhs_float = rhs_values[-1]
            ratio = lhs_float / rhs_float if rhs_float != 0.0 else np.inf
            print(
                f"step={int(step):d} "
                f"ell_t={float(grad_ell):.6g} "
                f"lhs={lhs_float:.6e} "
                f"rhs={rhs_float:.6e} "
                f"lhs/rhs={ratio:.6e}"
            )
        if print_mean_history:
            if fixed_history_for_print is None:
                print(f"step={int(step):d} adapt_mean={f_current:.6e}")
            else:
                history_idx = len(f_values) - 1
                if history_idx < len(fixed_history_for_print):
                    fixed_current = fixed_history_for_print[history_idx]
                    print(
                        f"step={int(step):d} "
                        f"fixed_mean={fixed_current:.6e} "
                        f"adapt_mean={f_current:.6e}"
                    )
                else:
                    print(f"step={int(step):d} adapt_mean={f_current:.6e}")
        prev_step = int(step)
        if stop_rel_tol is not None and rel_change is not None:
            if rel_change < stop_rel_tol:
                small_change_count += 1
            else:
                small_change_count = 0

            if small_change_count >= stop_patience:
                stopped_early = True
                if print_mean_history:
                    print(
                        f"Stopping adaptive at step={int(step):d}: "
                        f"relative_change={rel_change:.6e} "
                        f"< tol={stop_rel_tol:.6e} "
                        f"for {stop_patience:d} checkpoints"
                    )
                break

        prev_f = f_current

    if prev_step < n_steps and not stopped_early:
        x = run_segment(x, ell_schedule[prev_step:n_steps])

    f_final = F_with_precomputed_term_yy(x, means, covs, weights, eval_ell, term_yy_eval)
    actual_checkpoint_steps = checkpoint_steps[: len(f_values)]
    return (
        np.array(x),
        float(f_final),
        np.asarray(lhs_values, dtype=np.float64),
        np.asarray(rhs_values, dtype=np.float64),
        actual_checkpoint_steps,
        np.asarray(f_values, dtype=np.float64),
    )



def run_experiments(
    num_seeds,
    n_particles,
    fixed_n_steps,
    adapt_n_steps,
    fixed_step_size,
    ell_fixed,
    ell0,
    ell_min,
    decay,
    eval_ell=1.0,
    print_lhs_rhs=False,
    print_mean_history=False,
    adapt_stop_rel_tol=None,
    adapt_stop_patience=3,
    save_lhs_rhs_histories=False,
):
    fixed_finals = []
    adapt_finals = []
    fixed_histories = []
    adapt_histories = []
    adapt_lhs_histories = []
    adapt_rhs_histories = []
    last_fixed_particles = None
    last_adapt_particles = None
    last_adapt_lhs = None
    last_adapt_rhs = None
    last_adapt_checkpoint_steps = None
    history_steps = make_lhs_rhs_checkpoint_steps(min(fixed_n_steps, adapt_n_steps))

    means_jnp = jnp.array(means)
    covs_jnp = jnp.array(covs)
    weights_jnp = jnp.array(weights)

    for seed in range(num_seeds):
        rng = np.random.default_rng(seed)
        x0_np = rng.normal(loc=0.0, scale=1.0, size=(n_particles, d)).astype(np.float64)
        x0 = jnp.array(x0_np)

        ell_schedule_fixed = make_ell_schedule(
            n_steps=fixed_n_steps,
            mode="fixed",
            ell0=np.float64(ell_fixed),
            ell_min=np.float64(ell_min),
            decay=np.float64(decay),
        )
        ell_schedule_adapt = make_ell_schedule(
            n_steps=adapt_n_steps,
            mode="adaptive",
            ell0=np.float64(ell0),
            ell_min=np.float64(ell_min),
            decay=np.float64(decay),
        )

        fixed_particles, fixed_final, fixed_history = run_flow_fixed_with_history(
            x0=x0,
            means=means_jnp,
            covs=covs_jnp,
            weights=weights_jnp,
            ell_schedule=ell_schedule_fixed,
            step_size=np.float64(fixed_step_size),
            eval_ell=np.float64(eval_ell),
            checkpoint_steps=history_steps,
            stop_rel_tol=adapt_stop_rel_tol,
            stop_patience=adapt_stop_patience,
            print_stop=print_mean_history,
        )

        if seed == num_seeds - 1:
            (
                adapt_particles,
                adapt_final,
                last_adapt_lhs,
                last_adapt_rhs,
                last_adapt_checkpoint_steps,
                adapt_history,
            ) = run_flow_adaptive_with_lhs_rhs(
                x0=x0,
                means=means_jnp,
                covs=covs_jnp,
                weights=weights_jnp,
                ell_schedule=ell_schedule_adapt,
                ell_inf=np.float64(ell_min),
                eval_ell=np.float64(eval_ell),
                checkpoint_steps=make_lhs_rhs_checkpoint_steps(adapt_n_steps),
                print_lhs_rhs=print_lhs_rhs and seed == 0,
                fixed_history_for_print=fixed_history,
                print_mean_history=print_mean_history,
                stop_rel_tol=adapt_stop_rel_tol,
                stop_patience=adapt_stop_patience,
            )
        else:
            (
                adapt_particles,
                adapt_final,
                last_seed_lhs,
                last_seed_rhs,
                _last_seed_steps,
                adapt_history,
            ) = run_flow_adaptive_with_lhs_rhs(
                x0=x0,
                means=means_jnp,
                covs=covs_jnp,
                weights=weights_jnp,
                ell_schedule=ell_schedule_adapt,
                ell_inf=np.float64(ell_min),
                eval_ell=np.float64(eval_ell),
                checkpoint_steps=history_steps,
                print_lhs_rhs=print_lhs_rhs and seed == 0,
                fixed_history_for_print=fixed_history,
                print_mean_history=print_mean_history,
                stop_rel_tol=adapt_stop_rel_tol,
                stop_patience=adapt_stop_patience,
            )

        fixed_finals.append(fixed_final)
        adapt_finals.append(adapt_final)
        fixed_histories.append(fixed_history)
        adapt_histories.append(adapt_history[: history_steps.shape[0]])
        if save_lhs_rhs_histories:
            if seed == num_seeds - 1:
                adapt_lhs_histories.append(last_adapt_lhs[: history_steps.shape[0]])
                adapt_rhs_histories.append(last_adapt_rhs[: history_steps.shape[0]])
            else:
                adapt_lhs_histories.append(last_seed_lhs[: history_steps.shape[0]])
                adapt_rhs_histories.append(last_seed_rhs[: history_steps.shape[0]])
        last_fixed_particles = fixed_particles
        last_adapt_particles = adapt_particles

    max_fixed_history_len = max(history.shape[0] for history in fixed_histories)
    fixed_histories_padded = np.full(
        (len(fixed_histories), max_fixed_history_len),
        np.nan,
        dtype=np.float64,
    )
    for history_idx, history in enumerate(fixed_histories):
        fixed_histories_padded[history_idx, : history.shape[0]] = history

    max_adapt_history_len = max(history.shape[0] for history in adapt_histories)
    adapt_histories_padded = np.full(
        (len(adapt_histories), max_adapt_history_len),
        np.nan,
        dtype=np.float64,
    )
    for history_idx, history in enumerate(adapt_histories):
        adapt_histories_padded[history_idx, : history.shape[0]] = history
    fixed_history_mean = np.nanmean(fixed_histories_padded, axis=0)
    adapt_history_mean = np.nanmean(adapt_histories_padded, axis=0)
    fixed_history_std = np.nanstd(fixed_histories_padded, axis=0)
    adapt_history_std = np.nanstd(adapt_histories_padded, axis=0)
    fixed_history_count = np.sum(~np.isnan(fixed_histories_padded), axis=0)
    adapt_history_count = np.sum(~np.isnan(adapt_histories_padded), axis=0)
    fixed_history_se = fixed_history_std / np.sqrt(np.maximum(fixed_history_count, 1))
    adapt_history_se = adapt_history_std / np.sqrt(np.maximum(adapt_history_count, 1))

    fixed_finals = np.array(fixed_finals, dtype=np.float64)
    adapt_finals = np.array(adapt_finals, dtype=np.float64)

    results = {
        "fixed_finals": fixed_finals,
        "adapt_finals": adapt_finals,
        "fixed_mean": float(np.mean(fixed_finals)),
        "fixed_std": float(np.std(fixed_finals)),
        "fixed_se": float(np.std(fixed_finals) / np.sqrt(max(len(fixed_finals), 1))),
        "adapt_mean": float(np.mean(adapt_finals)),
        "adapt_std": float(np.std(adapt_finals)),
        "adapt_se": float(np.std(adapt_finals) / np.sqrt(max(len(adapt_finals), 1))),
        "last_fixed_particles": last_fixed_particles,
        "last_adapt_particles": last_adapt_particles,
        "last_adapt_lhs": last_adapt_lhs,
        "last_adapt_rhs": last_adapt_rhs,
        "last_adapt_checkpoint_steps": last_adapt_checkpoint_steps,
        "history_steps": history_steps,
        "fixed_history_steps": history_steps[: fixed_history_mean.shape[0]],
        "adapt_history_steps": history_steps[: adapt_history_mean.shape[0]],
        "fixed_histories": fixed_histories_padded,
        "adapt_histories": adapt_histories_padded,
        "fixed_history_mean": fixed_history_mean,
        "adapt_history_mean": adapt_history_mean,
        "fixed_history_std": fixed_history_std,
        "adapt_history_std": adapt_history_std,
        "fixed_history_se": fixed_history_se,
        "adapt_history_se": adapt_history_se,
        "fixed_history_count": fixed_history_count,
        "adapt_history_count": adapt_history_count,
        "save_lhs_rhs_histories": np.asarray(save_lhs_rhs_histories, dtype=np.bool_),
    }

    if save_lhs_rhs_histories:
        max_lhs_history_len = max(history.shape[0] for history in adapt_lhs_histories)
        adapt_lhs_histories_padded = np.full(
            (len(adapt_lhs_histories), max_lhs_history_len),
            np.nan,
            dtype=np.float64,
        )
        adapt_rhs_histories_padded = np.full(
            (len(adapt_rhs_histories), max_lhs_history_len),
            np.nan,
            dtype=np.float64,
        )
        for history_idx, history in enumerate(adapt_lhs_histories):
            adapt_lhs_histories_padded[history_idx, : history.shape[0]] = history
        for history_idx, history in enumerate(adapt_rhs_histories):
            adapt_rhs_histories_padded[history_idx, : history.shape[0]] = history

        adapt_lhs_mean = np.nanmean(adapt_lhs_histories_padded, axis=0)
        adapt_rhs_mean = np.nanmean(adapt_rhs_histories_padded, axis=0)
        adapt_lhs_std = np.nanstd(adapt_lhs_histories_padded, axis=0)
        adapt_rhs_std = np.nanstd(adapt_rhs_histories_padded, axis=0)
        adapt_lhs_count = np.sum(~np.isnan(adapt_lhs_histories_padded), axis=0)
        adapt_rhs_count = np.sum(~np.isnan(adapt_rhs_histories_padded), axis=0)

        results.update(
            {
                "adapt_lhs_checkpoint_steps": history_steps[: adapt_lhs_mean.shape[0]],
                "adapt_lhs_histories": adapt_lhs_histories_padded,
                "adapt_rhs_histories": adapt_rhs_histories_padded,
                "adapt_lhs_mean": adapt_lhs_mean,
                "adapt_rhs_mean": adapt_rhs_mean,
                "adapt_lhs_std": adapt_lhs_std,
                "adapt_rhs_std": adapt_rhs_std,
                "adapt_lhs_se": adapt_lhs_std / np.sqrt(np.maximum(adapt_lhs_count, 1)),
                "adapt_rhs_se": adapt_rhs_std / np.sqrt(np.maximum(adapt_rhs_count, 1)),
                "adapt_lhs_count": adapt_lhs_count,
                "adapt_rhs_count": adapt_rhs_count,
            }
        )

    return results


def save_results(results, output_path):
    np.savez_compressed(output_path, **results)


def load_results(input_path):
    with np.load(input_path) as data:
        return {key: data[key] for key in data.files}


if __name__ == "__main__":
    results_path = "results_n10f.npz"

    results = run_experiments(
        num_seeds=10,
        n_particles=10,
        fixed_n_steps=47000, 
        adapt_n_steps=47000,
        fixed_step_size=0.01,
        ell_fixed=0.1,
        ell0=10.0,
        ell_min=0.1,
        decay=0.9999,
        eval_ell=0.1,
        print_lhs_rhs=True,
        print_mean_history=True,
        adapt_stop_rel_tol=1e-6,
        adapt_stop_patience=3,
        save_lhs_rhs_histories=False,
    )

    save_results(results, results_path)

    print("Fixed final mean:", results["fixed_mean"])
    print("Adaptive final mean:", results["adapt_mean"])
    print("Saved results to:", results_path)

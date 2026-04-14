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
):
    fixed_finals = []
    adapt_finals = []
    last_fixed_particles = None
    last_adapt_particles = None

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

        fixed_particles, fixed_final = run_flow_fixed(
            x0=x0,
            means=means_jnp,
            covs=covs_jnp,
            weights=weights_jnp,
            ell_schedule=ell_schedule_fixed,
            step_size=np.float64(fixed_step_size),
            eval_ell=np.float64(eval_ell),
        )
        
        adapt_particles, adapt_final = run_flow_adaptive(
            x0=x0,
            means=means_jnp,
            covs=covs_jnp,
            weights=weights_jnp,
            ell_schedule=ell_schedule_adapt,
            eval_ell=np.float64(eval_ell),
        )

        fixed_finals.append(fixed_final)
        adapt_finals.append(adapt_final)
        last_fixed_particles = fixed_particles
        last_adapt_particles = adapt_particles

    return {
        "fixed_finals": np.array(fixed_finals, dtype=np.float64),
        "adapt_finals": np.array(adapt_finals, dtype=np.float64),
        "fixed_mean": float(np.mean(fixed_finals)),
        "fixed_std": float(np.std(fixed_finals)),
        "adapt_mean": float(np.mean(adapt_finals)),
        "adapt_std": float(np.std(adapt_finals)),
        "last_fixed_particles": last_fixed_particles,
        "last_adapt_particles": last_adapt_particles,
    }


def save_results(results, output_path):
    np.savez_compressed(output_path, **results)


def load_results(input_path):
    with np.load(input_path) as data:
        return {key: data[key] for key in data.files}


if __name__ == "__main__":
    results_path = "results_n300.npz"

    results = run_experiments(
        num_seeds=10,
        n_particles=300,
        fixed_n_steps=100000,
        adapt_n_steps=1000000,
        fixed_step_size=1.0,
        ell_fixed=0.1,
        ell0=10.0,
        ell_min=0.1,
        decay=0.9985,
        eval_ell=0.1,
    )

    save_results(results, results_path)

    print("Fixed final mean:", results["fixed_mean"])
    print("Adaptive final mean:", results["adapt_mean"])
    print("Saved results to:", results_path)

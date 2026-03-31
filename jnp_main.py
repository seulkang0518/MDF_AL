import numpy as np

import jax
import jax.numpy as jnp
from jax import lax


# ============================================================
# 1. Target: 10-component 2D mixture of Gaussians
# ============================================================
k = 10
radius = 4.0
d = 2

angles = np.linspace(0, 2 * np.pi, k, endpoint=False)

means = np.stack([
    radius * np.cos(angles),
    radius * np.sin(angles)
], axis=1).astype(np.float32)

covs = np.array([
    0.15 * np.eye(d, dtype=np.float32) for _ in range(k)
], dtype=np.float32)

weights = np.ones(k, dtype=np.float32) / k


def sample_mog(n, means, covs, weights, seed=None):
    rng = np.random.default_rng(seed)
    k, d = means.shape
    comp_ids = rng.choice(k, size=n, p=weights)
    x = np.zeros((n, d), dtype=np.float32)

    for j in range(k):
        idx = np.where(comp_ids == j)[0]
        if len(idx) > 0:
            x[idx] = rng.multivariate_normal(
                mean=means[j],
                cov=covs[j],
                size=len(idx)
            ).astype(np.float32)

    return x


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


def F(x, y, ell):
    """
    0.5 * unbiased empirical MMD^2 between x and y
    """
    n = x.shape[0]
    m = y.shape[0]

    kxx = gaussian_kernel(x, x, ell)
    kyy = gaussian_kernel(y, y, ell)
    kxy = gaussian_kernel(x, y, ell)

    kxx = kxx * (1.0 - jnp.eye(n, dtype=x.dtype))
    kyy = kyy * (1.0 - jnp.eye(m, dtype=y.dtype))

    term_xx = jnp.sum(kxx) / (n * (n - 1))
    term_yy = jnp.sum(kyy) / (m * (m - 1))
    term_xy = 2.0 * jnp.mean(kxy)

    return 0.5 * (term_xx + term_yy - term_xy)


def witness_gradient(x, y, ell):
    """
    Approximates ∇f_{P,Q}(x_i) for each particle x_i.
    x: (n, d)
    y: (m, d)
    returns: (n, d)
    """
    diff_xx = x[:, None, :] - x[None, :, :]   # (n, n, d)
    sq_xx = jnp.sum(diff_xx**2, axis=2)
    k_xx = jnp.exp(-0.5 * sq_xx / (ell**2))

    diff_yx = y[:, None, :] - x[None, :, :]   # (m, n, d)
    sq_yx = jnp.sum(diff_yx**2, axis=2)
    k_yx = jnp.exp(-0.5 * sq_yx / (ell**2))

    grad_emp = jnp.mean(k_xx[:, :, None] * (-diff_xx) / (ell**2), axis=1)
    grad_tar = jnp.mean(k_yx[:, :, None] * diff_yx / (ell**2), axis=0)

    return grad_emp - grad_tar


# ============================================================
# 3. Lengthscale schedules
# ============================================================
def make_ell_schedule(n_steps, mode, ell0, ell_min, decay):
    ts = jnp.arange(n_steps, dtype=jnp.float32)

    if mode == "fixed":
        return jnp.full((n_steps,), ell0, dtype=jnp.float32)
    elif mode == "adaptive":
        return jnp.maximum(ell_min, ell0 * (decay ** ts))
    else:
        raise ValueError("mode must be 'fixed' or 'adaptive'")


# ============================================================
# 4. JAX GF loop
# ============================================================
@jax.jit
def mmd_gf_jax_light(x0, y, ell_schedule, step_size, eval_ell):
    """
    Runs particle GF and returns final particles + mmd history.
    Does not store the full trajectory, to save memory.
    """
    def one_step(x, ell):
        grad = witness_gradient(x, y, ell)
        x_new = x - step_size * grad
        f_val = F(x_new, y, eval_ell)
        return x_new, f_val

    x_final, mmd_hist = lax.scan(one_step, x0, ell_schedule)
    return x_final, mmd_hist


# Optional: version that stores trajectory for one run only
@jax.jit
def mmd_gf_jax_with_traj(x0, y, ell_schedule, step_size, eval_ell):
    def one_step(x, ell):
        grad = witness_gradient(x, y, ell)
        x_new = x - step_size * grad
        f_val = F(x_new, y, eval_ell)
        return x_new, (x_new, f_val, ell)

    x_final, (traj, mmd_hist, ell_hist) = lax.scan(one_step, x0, ell_schedule)
    return x_final, traj, mmd_hist, ell_hist


# ============================================================
# 5. Multi-seed experiment
# ============================================================
def run_experiments(
    num_seeds,
    n_particles,
    n_target,
    n_steps,
    step_size,
    ell_fixed,
    ell0,
    ell_min,
    decay,
    eval_ell=1.0,
):
    fixed_hists = []
    adapt_hists = []

    last_x0 = None
    last_y = None
    last_fixed_particles = None
    last_adapt_particles = None
    last_fixed_ell_hist = None
    last_adapt_ell_hist = None

    for seed in range(num_seeds):
        rng = np.random.default_rng(seed)

        # target samples
        y_np = sample_mog(n_target, means, covs, weights, seed=seed)

        # same init for both methods within the same seed
        x0_np = rng.normal(loc=0.0, scale=1.0, size=(n_particles, d)).astype(np.float32)

        # convert once
        y = jnp.array(y_np)
        x0 = jnp.array(x0_np)

        ell_schedule_fixed = make_ell_schedule(
            n_steps=n_steps,
            mode="fixed",
            ell0=np.float32(ell_fixed),
            ell_min=np.float32(ell_min),
            decay=np.float32(decay),
        )

        ell_schedule_adapt = make_ell_schedule(
            n_steps=n_steps,
            mode="adaptive",
            ell0=np.float32(ell0),
            ell_min=np.float32(ell_min),
            decay=np.float32(decay),
        )

        fixed_particles, fixed_hist = mmd_gf_jax_light(
            x0=x0,
            y=y,
            ell_schedule=ell_schedule_fixed,
            step_size=np.float32(step_size),
            eval_ell=np.float32(eval_ell),
        )

        adapt_particles, adapt_hist = mmd_gf_jax_light(
            x0=x0,
            y=y,
            ell_schedule=ell_schedule_adapt,
            step_size=np.float32(step_size),
            eval_ell=np.float32(eval_ell),
        )

        fixed_hists.append(np.array(fixed_hist))
        adapt_hists.append(np.array(adapt_hist))

        # keep one example run for plotting particles
        last_x0 = x0_np
        last_y = y_np
        last_fixed_particles = np.array(fixed_particles)
        last_adapt_particles = np.array(adapt_particles)
        last_fixed_ell_hist = np.array(ell_schedule_fixed)
        last_adapt_ell_hist = np.array(ell_schedule_adapt)

    fixed_hists = np.array(fixed_hists)   # (num_seeds, n_steps)
    adapt_hists = np.array(adapt_hists)   # (num_seeds, n_steps)

    return {
        "fixed_hists": fixed_hists,
        "adapt_hists": adapt_hists,
        "fixed_mean": fixed_hists.mean(axis=0),
        "fixed_std": fixed_hists.std(axis=0),
        "adapt_mean": adapt_hists.mean(axis=0),
        "adapt_std": adapt_hists.std(axis=0),
        "last_x0": last_x0,
        "last_y": last_y,
        "last_fixed_particles": last_fixed_particles,
        "last_adapt_particles": last_adapt_particles,
        "last_fixed_ell_hist": last_fixed_ell_hist,
        "last_adapt_ell_hist": last_adapt_ell_hist,
    }


# ============================================================
# 6. Settings
# ============================================================
num_seeds = 30
n_particles = 300
n_target = 3000
n_steps = 1000
step_size = 0.05

ell_fixed = 0.1
ell0 = 10.0
ell_min = 0.1
decay = 0.995
eval_ell = 1.0


# ============================================================
# 7. Run experiment
# ============================================================
results = run_experiments(
    num_seeds=num_seeds,
    n_particles=n_particles,
    n_target=n_target,
    n_steps=n_steps,
    step_size=step_size,
    ell_fixed=ell_fixed,
    ell0=ell0,
    ell_min=ell_min,
    decay=decay,
    eval_ell=eval_ell,
)

print("Fixed final mean:", results["fixed_mean"][-1])
print("Adaptive final mean:", results["adapt_mean"][-1])


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 18
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts, mathrsfs, amssymb}')

# plt.rc('font', family='Arial', size=12)
plt.rc('axes', titlesize=32, labelsize=32, grid=True)
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=32, frameon=False)
plt.rc('xtick', labelsize=26, direction='in')
plt.rc('ytick', labelsize=26, direction='in')
plt.rcParams['lines.markersize'] = 14
plt.rc('figure', figsize=(6, 4), dpi=100)
FIGSIZE=(6, 4)
DPI = 100
LOCAL_PLOT_RC = {
    "text.usetex": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
}


def _load_npz_dict(npz_path):
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def _sample_target_mog(num_samples, seed=0, radius=2.0, std=0.2, k=8):
    rng = np.random.default_rng(seed)
    angles = 2.0 * np.pi * np.arange(k, dtype=np.float64) / k
    means = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles)],
        axis=1,
    )
    component_ids = rng.integers(0, k, size=num_samples)
    return means[component_ids] + rng.normal(scale=std, size=(num_samples, 2))


def make_comparison_plot(
    target_samples,
    fixed_samples,
    adaptive_samples,
    fixed_mmd,
    adaptive_mmd,
    output_path,
):
    all_samples = np.vstack([target_samples, fixed_samples, adaptive_samples])
    mins = all_samples.min(axis=0)
    maxs = all_samples.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = np.max(maxs - mins)
    half_width = 0.55 * span
    x_min, x_max = center[0] - half_width, center[0] + half_width
    y_min, y_max = center[1] - half_width, center[1] + half_width

    heatmap, xedges, yedges = np.histogram2d(
        target_samples[:, 0],
        target_samples[:, 1],
        bins=80,
        range=[[x_min, x_max], [y_min, y_max]],
        density=True,
    )

    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        ax.imshow(
            heatmap.T,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            origin="lower",
            cmap="Blues",
            alpha=0.6,
            aspect="auto",
        )
        ax.scatter(
            fixed_samples[:, 0],
            fixed_samples[:, 1],
            s=7,
            c="#F17C67",
            alpha=0.55,
            edgecolors="none",
            label=f"Fixed ({fixed_mmd:.4f})",
        )
        ax.scatter(
            adaptive_samples[:, 0],
            adaptive_samples[:, 1],
            s=7,
            c="#B23AEE",
            alpha=0.45,
            edgecolors="none",
            label=f"Adaptive ({adaptive_mmd:.4f})",
        )
        ax.set_title("Target Density vs Fixed vs Adaptive", pad=8)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, alpha=0.15, linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)


def make_mmd_vs_n_plot(ns, fixed_mmd, adaptive_mmd, output_path):
    ns = np.asarray(ns, dtype=float)
    fixed_mmd = np.asarray(fixed_mmd, dtype=float)
    adaptive_mmd = np.asarray(adaptive_mmd, dtype=float)

    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        ax.plot(ns, fixed_mmd, marker="o", color="#F17C67", label="Fixed")
        ax.plot(ns, adaptive_mmd, marker="o", color="#B23AEE", label="Adaptive")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("MMD")
        ax.set_title("MMD vs n", pad=8)
        ax.grid(True, which="major", alpha=0.15, linewidth=0.8)
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)


def make_mmd_vs_iteration_plot(fixed_mmd, adaptive_mmd, output_path):
    fixed_mmd = np.asarray(fixed_mmd, dtype=float)
    adaptive_mmd = np.asarray(adaptive_mmd, dtype=float)
    fixed_steps = np.arange(1, fixed_mmd.shape[0] + 1, dtype=float)
    adaptive_steps = np.arange(1, adaptive_mmd.shape[0] + 1, dtype=float)

    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        ax.plot(fixed_steps, fixed_mmd, color="#F17C67", label="Fixed")
        ax.plot(adaptive_steps, adaptive_mmd, color="#B23AEE", label="Adaptive")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MMD")
        ax.set_title("MMD vs Iteration", pad=8)
        ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)


def make_lhs_rhs_plot(steps, lhs, rhs, output_path):
    steps = np.asarray(steps, dtype=float)
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)

    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        ax.plot(steps, lhs, color="#D62728", label="LHS")
        ax.plot(steps, rhs, color="#111111", label="RHS")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.set_title("LHS vs RHS", pad=8)
        ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)


def make_mmd_vs_iteration_plot_from_npz(npz_path, output_path):
    data = _load_npz_dict(npz_path)
    fixed_hists = np.asarray(data["fixed_hists"], dtype=float)
    adaptive_hists = np.asarray(data["adapt_hists"], dtype=float)
    fixed_mmd = np.sqrt(np.maximum(2.0 * fixed_hists.mean(axis=0), 0.0))
    adaptive_mmd = np.sqrt(np.maximum(2.0 * adaptive_hists.mean(axis=0), 0.0))
    make_mmd_vs_iteration_plot(fixed_mmd, adaptive_mmd, output_path)


def make_lhs_rhs_plot_from_npz(npz_path, output_path):
    data = _load_npz_dict(npz_path)
    steps = np.asarray(data["last_adapt_checkpoint_steps"], dtype=float)
    lhs = np.asarray(data["last_adapt_lhs"], dtype=float)
    rhs = np.asarray(data["last_adapt_rhs"], dtype=float)
    make_lhs_rhs_plot(steps, lhs, rhs, output_path)


def make_comparison_plot_from_npz(
    npz_path,
    output_path,
    num_target_samples=5000,
    target_seed=0,
    radius=2.0,
    std=0.2,
    k=8,
):
    data = _load_npz_dict(npz_path)
    target_samples = _sample_target_mog(
        num_samples=num_target_samples,
        seed=target_seed,
        radius=radius,
        std=std,
        k=k,
    )
    fixed_samples = np.asarray(data["last_fixed_particles"], dtype=float)
    adaptive_samples = np.asarray(data["last_adapt_particles"], dtype=float)
    fixed_mmd = np.sqrt(max(2.0 * float(data["fixed_mean"]), 0.0))
    adaptive_mmd = np.sqrt(max(2.0 * float(data["adapt_mean"]), 0.0))
    make_comparison_plot(
        target_samples=target_samples,
        fixed_samples=fixed_samples,
        adaptive_samples=adaptive_samples,
        fixed_mmd=fixed_mmd,
        adaptive_mmd=adaptive_mmd,
        output_path=output_path,
    )


def make_mmd_vs_n_plot_from_npz(npz_paths, ns, output_path):
    fixed_mmd = []
    adaptive_mmd = []

    for npz_path in npz_paths:
        data = _load_npz_dict(npz_path)
        fixed_mmd.append(np.sqrt(max(2.0 * float(data["fixed_mean"]), 0.0)))
        adaptive_mmd.append(np.sqrt(max(2.0 * float(data["adapt_mean"]), 0.0)))

    make_mmd_vs_n_plot(
        ns=np.asarray(ns, dtype=float),
        fixed_mmd=np.asarray(fixed_mmd, dtype=float),
        adaptive_mmd=np.asarray(adaptive_mmd, dtype=float),
        output_path=output_path,
    )


def merge_four_figures(image_paths, output_path, titles=None):
    images = [plt.imread(path) for path in image_paths]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), dpi=150)
    axes = np.atleast_1d(axes)

    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")

    for ax in axes[len(images) :]:
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.04)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    comparison_npz = "/Users/sophiakang/Documents/GitHub/MDF_AL/results_n300.npz"
    mmd_vs_n_npz_paths = [
        "/Users/sophiakang/Documents/GitHub/MDF_AL/results_n10.npz",
        "/Users/sophiakang/Documents/GitHub/MDF_AL/results_n30.npz",
        "/Users/sophiakang/Documents/GitHub/MDF_AL/results_n100.npz",
        "/Users/sophiakang/Documents/GitHub/MDF_AL/results_n300.npz",
    ]
    mmd_vs_n_ns = [10, 30, 100, 300]
    mmd_vs_iteration_npz = "/Users/sophiakang/Documents/GitHub/MDF_AL/mmd_iteration_100.npz"
    lhs_rhs_npz = "/Users/sophiakang/Documents/GitHub/MDF_AL/lhs_rhs.npz"

    comparison_out = "/Users/sophiakang/Documents/GitHub/MDF_AL/mog_comparison.png"
    mmd_vs_n_out = "/Users/sophiakang/Documents/GitHub/MDF_AL/mmd_vs_n.png"
    mmd_vs_iteration_out = "/Users/sophiakang/Documents/GitHub/MDF_AL/mmd_vs_iteration.png"
    lhs_rhs_out = "/Users/sophiakang/Documents/GitHub/MDF_AL/lhs_rhs.png"
    merged_out = "/Users/sophiakang/Documents/GitHub/MDF_AL/all_figures.png"

    make_comparison_plot_from_npz(comparison_npz, comparison_out)
    make_mmd_vs_n_plot_from_npz(mmd_vs_n_npz_paths, mmd_vs_n_ns, mmd_vs_n_out)
    make_mmd_vs_iteration_plot_from_npz(mmd_vs_iteration_npz, mmd_vs_iteration_out)
    make_lhs_rhs_plot_from_npz(lhs_rhs_npz, lhs_rhs_out)

    merge_four_figures(
        [comparison_out, mmd_vs_n_out, mmd_vs_iteration_out, lhs_rhs_out],
        merged_out,
    )

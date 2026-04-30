from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FIGSIZE = (6, 4)
DPI = 100
SUMMARY_FIGSIZE = (24, 4.8)
SUMMARY_DPI = 150
TITLE_SIZE = 24
LABEL_SIZE = 20
TICK_SIZE = 16
LEGEND_SIZE = 32
MARKER_SIZE = 4.5
LINE_WIDTH = 2.0
TITLE_PAD = 8
FOUR_PANEL_LEGEND_SIZE = 28

COLORS = {
    "mmd": "#d95f02",
    "smmd": "#1b9e77",
    "svgd": "#7570b3",
    "sgd": "#1b9e77",
    "sgd_dark": "#0f6f54",
    "natural": "#7570b3",
    "natural_dark": "#5b5696",
    "pgd": "#d95f02",
    "pgd_dark": "#D62728",
    "lhs": "#D62728",
    "rhs": "#111111",
    "truth": "#f4b400",
}

BASE_PLOT_RC = {
    "axes.grid": True,
    "font.family": "DeJavu Serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "legend.frameon": False,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": LINE_WIDTH,
    "lines.markersize": MARKER_SIZE,
    "figure.figsize": FIGSIZE,
    "figure.dpi": DPI,
}
LOCAL_PLOT_RC = dict(BASE_PLOT_RC)
plt.rcParams.update(BASE_PLOT_RC)


def _load_npz_dict(npz_path):
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def _save_figure(fig, output_path=None, **savefig_kwargs):
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, **savefig_kwargs)
    return fig


def _sample_target_mog(num_samples, seed=0, radius=2.0, std=0.2, k=8):
    rng = np.random.default_rng(seed)
    angles = 2.0 * np.pi * np.arange(k, dtype=np.float64) / k
    means = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    component_ids = rng.integers(0, k, size=num_samples)
    return means[component_ids] + rng.normal(scale=std, size=(num_samples, 2))


def _comparison_plot_limits(target_samples, fixed_samples, adaptive_samples):
    all_samples = np.vstack([target_samples, fixed_samples, adaptive_samples])
    mins = all_samples.min(axis=0)
    maxs = all_samples.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = np.max(maxs - mins)
    half_width = 0.55 * span
    return (
        center[0] - half_width,
        center[0] + half_width,
        center[1] - half_width,
        center[1] + half_width,
    )


def _draw_comparison_plot(ax, target_samples, fixed_samples, adaptive_samples, show_legend=True):
    x_min, x_max, y_min, y_max = _comparison_plot_limits(
        target_samples,
        fixed_samples,
        adaptive_samples,
    )
    ax.scatter(
        fixed_samples[:, 0],
        fixed_samples[:, 1],
        s=7,
        c=COLORS["mmd"],
        alpha=0.55,
        edgecolors="none",
        label="Fixed",
    )
    ax.scatter(
        adaptive_samples[:, 0],
        adaptive_samples[:, 1],
        s=7,
        c=COLORS["smmd"],
        alpha=0.45,
        edgecolors="none",
        label="Adaptive",
    )
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.15, linewidth=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    if show_legend:
        ax.legend(loc="upper right", frameon=True)


def _mmd_band_from_mean_se(mean, se=None, factor=2.0, se_scale=1.0):
    mean = np.asarray(mean, dtype=float)
    mmd = np.sqrt(np.maximum(factor * mean, 0.0))
    if se is None:
        return mmd, None

    se = se_scale * np.asarray(se, dtype=float)
    lower = np.sqrt(np.maximum(factor * (mean - se), 0.0))
    upper = np.sqrt(np.maximum(factor * (mean + se), 0.0))
    return mmd, (lower, upper)


def _mmd_band_from_histories(histories, factor=2.0, se_scale=1.0):
    histories = np.asarray(histories, dtype=float)
    mmd_histories = np.sqrt(np.maximum(factor * histories, 0.0))
    mmd = np.nanmean(mmd_histories, axis=0)
    valid_counts = np.maximum(np.sum(~np.isnan(mmd_histories), axis=0), 1)
    se = se_scale * np.nanstd(mmd_histories, axis=0) / np.sqrt(valid_counts)
    return mmd, (np.maximum(mmd - se, 0.0), mmd + se)


def _draw_mmd_vs_n_plot(
    ax,
    ns,
    fixed_mmd,
    adaptive_mmd,
    fixed_band=None,
    adaptive_band=None,
    show_legend=True,
    show_ylabel=True,
):
    ns = np.asarray(ns, dtype=float)
    fixed_mmd = np.asarray(fixed_mmd, dtype=float)
    adaptive_mmd = np.asarray(adaptive_mmd, dtype=float)

    if fixed_band is not None:
        ax.fill_between(ns, fixed_band[0], fixed_band[1], color=COLORS["mmd"], alpha=0.22, linewidth=0)
    if adaptive_band is not None:
        ax.fill_between(ns, adaptive_band[0], adaptive_band[1], color=COLORS["smmd"], alpha=0.22, linewidth=0)

    ax.plot(ns, fixed_mmd, marker="o", color=COLORS["mmd"], label="Fixed", zorder=3)
    ax.plot(ns, adaptive_mmd, marker="o", color=COLORS["smmd"], label="Adaptive", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    if show_ylabel:
        ax.set_ylabel("MMD")
    ax.grid(True, which="major", alpha=0.15, linewidth=0.8)
    if show_legend:
        ax.legend(frameon=True)


def _draw_mmd_vs_iteration_plot(
    ax,
    fixed_mmd,
    adaptive_mmd,
    fixed_steps=None,
    adaptive_steps=None,
    fixed_band=None,
    adaptive_band=None,
    show_legend=True,
    show_ylabel=True,
):
    fixed_mmd = np.asarray(fixed_mmd, dtype=float)
    adaptive_mmd = np.asarray(adaptive_mmd, dtype=float)
    if fixed_steps is None:
        fixed_steps = np.arange(1, fixed_mmd.shape[0] + 1, dtype=float)
    if adaptive_steps is None:
        adaptive_steps = np.arange(1, adaptive_mmd.shape[0] + 1, dtype=float)
    fixed_steps = np.asarray(fixed_steps, dtype=float)
    adaptive_steps = np.asarray(adaptive_steps, dtype=float)

    if fixed_band is not None:
        ax.fill_between(fixed_steps, fixed_band[0], fixed_band[1], color=COLORS["mmd"], alpha=0.22, linewidth=0)
    if adaptive_band is not None:
        ax.fill_between(
            adaptive_steps,
            adaptive_band[0],
            adaptive_band[1],
            color=COLORS["smmd"],
            alpha=0.22,
            linewidth=0,
        )

    ax.plot(fixed_steps, fixed_mmd, color=COLORS["mmd"], label="Fixed", zorder=3)
    ax.plot(adaptive_steps, adaptive_mmd, color=COLORS["smmd"], label="Adaptive", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    if show_ylabel:
        ax.set_ylabel("MMD")
    ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
    if show_legend:
        ax.legend(frameon=True)


def _draw_lhs_rhs_plot(ax, steps, lhs, rhs, show_legend=True, show_ylabel=True):
    steps = np.asarray(steps, dtype=float)
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    ax.plot(steps, lhs, color=COLORS["lhs"], label="LHS")
    ax.plot(steps, rhs, color=COLORS["rhs"], label="RHS")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    if show_ylabel:
        ax.set_ylabel("Value")
    ax.set_title("LHS vs RHS", pad=TITLE_PAD)
    ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
    if show_legend:
        ax.legend(frameon=True)


def make_comparison_plot(
    target_samples,
    fixed_samples,
    adaptive_samples,
    fixed_mmd=None,
    adaptive_mmd=None,
    output_path=None,
):
    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        _draw_comparison_plot(ax, target_samples, fixed_samples, adaptive_samples)
        fig.tight_layout()
        return _save_figure(fig, output_path)


def make_mmd_vs_n_plot(ns, fixed_mmd, adaptive_mmd, output_path=None, fixed_band=None, adaptive_band=None):
    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        _draw_mmd_vs_n_plot(
            ax,
            ns,
            fixed_mmd,
            adaptive_mmd,
            fixed_band=fixed_band,
            adaptive_band=adaptive_band,
        )
        fig.tight_layout()
        return _save_figure(fig, output_path)


def make_mmd_vs_iteration_plot(
    fixed_mmd,
    adaptive_mmd,
    output_path=None,
    fixed_steps=None,
    adaptive_steps=None,
    fixed_band=None,
    adaptive_band=None,
):
    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        _draw_mmd_vs_iteration_plot(
            ax,
            fixed_mmd,
            adaptive_mmd,
            fixed_steps=fixed_steps,
            adaptive_steps=adaptive_steps,
            fixed_band=fixed_band,
            adaptive_band=adaptive_band,
        )
        fig.tight_layout()
        return _save_figure(fig, output_path)


def make_lhs_rhs_plot(steps, lhs, rhs, output_path=None):
    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        _draw_lhs_rhs_plot(ax, steps, lhs, rhs)
        fig.tight_layout()
        return _save_figure(fig, output_path)


def _mmd_iteration_from_npz_data(data, se_scale=1.0):
    if "fixed_hists" in data and "adapt_hists" in data:
        fixed_mmd, fixed_band = _mmd_band_from_histories(data["fixed_hists"], se_scale=se_scale)
        adaptive_mmd, adaptive_band = _mmd_band_from_histories(data["adapt_hists"], se_scale=se_scale)
        fixed_steps = np.arange(1, fixed_mmd.shape[0] + 1, dtype=float)
        adaptive_steps = np.arange(1, adaptive_mmd.shape[0] + 1, dtype=float)
        return fixed_steps, fixed_mmd, fixed_band, adaptive_steps, adaptive_mmd, adaptive_band

    if "fixed_histories" in data and "adapt_histories" in data:
        fixed_mmd, fixed_band = _mmd_band_from_histories(data["fixed_histories"], se_scale=se_scale)
        adaptive_mmd, adaptive_band = _mmd_band_from_histories(data["adapt_histories"], se_scale=se_scale)
        fixed_steps = np.asarray(data.get("fixed_history_steps", np.arange(1, fixed_mmd.shape[0] + 1)), dtype=float)
        adaptive_steps = np.asarray(data.get("adapt_history_steps", np.arange(1, adaptive_mmd.shape[0] + 1)), dtype=float)
        return fixed_steps, fixed_mmd, fixed_band, adaptive_steps, adaptive_mmd, adaptive_band

    fixed_steps = np.asarray(data["fixed_history_steps"], dtype=float)
    adaptive_steps = np.asarray(data["adapt_history_steps"], dtype=float)
    fixed_se = data["fixed_history_se"] if "fixed_history_se" in data else None
    adaptive_se = data["adapt_history_se"] if "adapt_history_se" in data else None
    fixed_mmd, fixed_band = _mmd_band_from_mean_se(data["fixed_history_mean"], fixed_se, se_scale=se_scale)
    adaptive_mmd, adaptive_band = _mmd_band_from_mean_se(data["adapt_history_mean"], adaptive_se, se_scale=se_scale)
    return fixed_steps, fixed_mmd, fixed_band, adaptive_steps, adaptive_mmd, adaptive_band


def make_mmd_vs_iteration_plot_from_npz(npz_path, output_path=None, se_scale=1.0):
    data = _load_npz_dict(npz_path)
    fixed_steps, fixed_mmd, fixed_band, adaptive_steps, adaptive_mmd, adaptive_band = _mmd_iteration_from_npz_data(
        data,
        se_scale=se_scale,
    )
    return make_mmd_vs_iteration_plot(
        fixed_mmd,
        adaptive_mmd,
        output_path=output_path,
        fixed_steps=fixed_steps,
        adaptive_steps=adaptive_steps,
        fixed_band=fixed_band,
        adaptive_band=adaptive_band,
    )


def make_lhs_rhs_plot_from_npz(npz_path, output_path=None):
    data = _load_npz_dict(npz_path)
    steps, lhs, rhs = _get_lhs_rhs_series(data)
    return make_lhs_rhs_plot(steps, lhs, rhs, output_path=output_path)


def make_comparison_plot_from_npz(
    npz_path,
    output_path=None,
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
    return make_comparison_plot(
        target_samples=target_samples,
        fixed_samples=fixed_samples,
        adaptive_samples=adaptive_samples,
        fixed_mmd=fixed_mmd,
        adaptive_mmd=adaptive_mmd,
        output_path=output_path,
    )


def make_mmd_vs_n_plot_from_npz(npz_paths, ns, output_path=None, se_scale=1.0):
    fixed_mmd = []
    adaptive_mmd = []
    fixed_lower = []
    fixed_upper = []
    adaptive_lower = []
    adaptive_upper = []
    have_fixed_band = True
    have_adaptive_band = True

    for npz_path in npz_paths:
        data = _load_npz_dict(npz_path)
        fixed_value, fixed_band = _mmd_band_from_mean_se(
            data["fixed_mean"],
            data["fixed_se"] if "fixed_se" in data else None,
            se_scale=se_scale,
        )
        adaptive_value, adaptive_band = _mmd_band_from_mean_se(
            data["adapt_mean"],
            data["adapt_se"] if "adapt_se" in data else None,
            se_scale=se_scale,
        )
        fixed_mmd.append(float(fixed_value))
        adaptive_mmd.append(float(adaptive_value))
        if fixed_band is None:
            have_fixed_band = False
        else:
            fixed_lower.append(float(fixed_band[0]))
            fixed_upper.append(float(fixed_band[1]))
        if adaptive_band is None:
            have_adaptive_band = False
        else:
            adaptive_lower.append(float(adaptive_band[0]))
            adaptive_upper.append(float(adaptive_band[1]))

    fixed_band = (np.asarray(fixed_lower), np.asarray(fixed_upper)) if have_fixed_band else None
    adaptive_band = (np.asarray(adaptive_lower), np.asarray(adaptive_upper)) if have_adaptive_band else None

    return make_mmd_vs_n_plot(
        ns=np.asarray(ns, dtype=float),
        fixed_mmd=np.asarray(fixed_mmd, dtype=float),
        adaptive_mmd=np.asarray(adaptive_mmd, dtype=float),
        output_path=output_path,
        fixed_band=fixed_band,
        adaptive_band=adaptive_band,
    )


def make_four_panel_figure(
    comparison_npz,
    mmd_vs_n_npz_paths,
    mmd_vs_n_ns,
    mmd_vs_iteration_npz,
    lhs_rhs_npz,
    output_path=None,
    se_scale=1.0,
):
    comparison_data = _load_npz_dict(comparison_npz)
    target_samples = _sample_target_mog(num_samples=5000, seed=0, radius=2.0, std=0.2, k=8)
    fixed_samples = np.asarray(comparison_data["last_fixed_particles"], dtype=float)
    adaptive_samples = np.asarray(comparison_data["last_adapt_particles"], dtype=float)

    mmd_vs_n_fixed = []
    mmd_vs_n_adaptive = []
    mmd_vs_n_fixed_lower = []
    mmd_vs_n_fixed_upper = []
    mmd_vs_n_adaptive_lower = []
    mmd_vs_n_adaptive_upper = []
    have_mmd_vs_n_fixed_band = True
    have_mmd_vs_n_adaptive_band = True
    for npz_path in mmd_vs_n_npz_paths:
        data = _load_npz_dict(npz_path)
        fixed_value, fixed_band = _mmd_band_from_mean_se(
            data["fixed_mean"],
            data["fixed_se"] if "fixed_se" in data else None,
            se_scale=se_scale,
        )
        adaptive_value, adaptive_band = _mmd_band_from_mean_se(
            data["adapt_mean"],
            data["adapt_se"] if "adapt_se" in data else None,
            se_scale=se_scale,
        )
        mmd_vs_n_fixed.append(float(fixed_value))
        mmd_vs_n_adaptive.append(float(adaptive_value))
        if fixed_band is None:
            have_mmd_vs_n_fixed_band = False
        else:
            mmd_vs_n_fixed_lower.append(float(fixed_band[0]))
            mmd_vs_n_fixed_upper.append(float(fixed_band[1]))
        if adaptive_band is None:
            have_mmd_vs_n_adaptive_band = False
        else:
            mmd_vs_n_adaptive_lower.append(float(adaptive_band[0]))
            mmd_vs_n_adaptive_upper.append(float(adaptive_band[1]))

    mmd_vs_n_fixed = np.asarray(mmd_vs_n_fixed, dtype=float)
    mmd_vs_n_adaptive = np.asarray(mmd_vs_n_adaptive, dtype=float)
    mmd_vs_n_ns = np.asarray(mmd_vs_n_ns, dtype=float)
    mmd_vs_n_fixed_band = (
        (np.asarray(mmd_vs_n_fixed_lower), np.asarray(mmd_vs_n_fixed_upper))
        if have_mmd_vs_n_fixed_band
        else None
    )
    mmd_vs_n_adaptive_band = (
        (np.asarray(mmd_vs_n_adaptive_lower), np.asarray(mmd_vs_n_adaptive_upper))
        if have_mmd_vs_n_adaptive_band
        else None
    )

    iteration_data = _load_npz_dict(mmd_vs_iteration_npz)
    (
        iteration_fixed_steps,
        iteration_fixed_mmd,
        iteration_fixed_band,
        iteration_adaptive_steps,
        iteration_adaptive_mmd,
        iteration_adaptive_band,
    ) = _mmd_iteration_from_npz_data(iteration_data, se_scale=se_scale)

    lhs_rhs_data = _load_npz_dict(lhs_rhs_npz)
    lhs_rhs_steps, lhs_values, rhs_values = _get_lhs_rhs_series(lhs_rhs_data)

    top_legend_handles = [
        Line2D([0], [0], color=COLORS["mmd"], marker="o", lw=2, label="Fixed"),
        Line2D([0], [0], color=COLORS["smmd"], marker="o", lw=2, label="Adaptive"),
    ]

    with plt.rc_context(LOCAL_PLOT_RC):
        fig, axes = plt.subplots(1, 4, figsize=(24, 5), dpi=100)
        ax_comparison, ax_mmd_n, ax_mmd_iter, ax_lhs_rhs = axes
        _draw_comparison_plot(ax_comparison, target_samples, fixed_samples, adaptive_samples, show_legend=False)
        _draw_mmd_vs_n_plot(
            ax_mmd_n,
            mmd_vs_n_ns,
            mmd_vs_n_fixed,
            mmd_vs_n_adaptive,
            fixed_band=mmd_vs_n_fixed_band,
            adaptive_band=mmd_vs_n_adaptive_band,
            show_legend=False,
        )
        _draw_mmd_vs_iteration_plot(
            ax_mmd_iter,
            iteration_fixed_mmd,
            iteration_adaptive_mmd,
            fixed_steps=iteration_fixed_steps,
            adaptive_steps=iteration_adaptive_steps,
            fixed_band=iteration_fixed_band,
            adaptive_band=iteration_adaptive_band,
            show_legend=False,
        )
        _draw_lhs_rhs_plot(ax_lhs_rhs, lhs_rhs_steps, lhs_values, rhs_values)
        fig.legend(
            handles=top_legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=2,
            fancybox=False,
            facecolor="white",
            fontsize=FOUR_PANEL_LEGEND_SIZE,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return _save_figure(fig, output_path, bbox_inches="tight")

def _gk_mmd_and_band_from_f_history(data, prefix, se_scale=1.96):
    mean_key = f"{prefix}_eval_history_mean"
    histories_key = f"{prefix}_eval_histories"
    se_key = f"{prefix}_eval_history_se"

    if histories_key in data:
        f_histories = np.asarray(data[histories_key], dtype=float)
        f_mean = np.nanmean(f_histories, axis=0)
        mmd = np.sqrt(np.maximum(f_mean, 0.0))
        mmd_histories = np.sqrt(np.maximum(f_histories, 0.0))
        valid_counts = np.maximum(np.sum(~np.isnan(mmd_histories), axis=0), 1)
        se = se_scale * np.nanstd(mmd_histories, axis=0) / np.sqrt(valid_counts)
        return mmd, (np.maximum(mmd - se, 0.0), mmd + se)

    f_mean = np.asarray(data[mean_key], dtype=float)
    mmd = np.sqrt(np.maximum(f_mean, 0.0))
    if se_key not in data:
        return mmd, None

    f_se = se_scale * np.asarray(data[se_key], dtype=float)
    lower = np.sqrt(np.maximum(f_mean - f_se, 0.0))
    upper = np.sqrt(np.maximum(f_mean + f_se, 0.0))
    return mmd, (lower, upper)


def _draw_gk_mmd_vs_iteration_plot(ax, data, show_legend=False, show_ylabel=True, se_scale=1.96):
    baseline_steps = np.asarray(data["baseline_history_steps"], dtype=float) + 1.0
    adaptive_steps = np.asarray(data["adaptive_history_steps"], dtype=float) + 1.0
    baseline_mmd, baseline_band = _gk_mmd_and_band_from_f_history(data, "baseline", se_scale=se_scale)
    adaptive_mmd, adaptive_band = _gk_mmd_and_band_from_f_history(data, "adaptive", se_scale=se_scale)

    if baseline_band is not None:
        ax.fill_between(
            baseline_steps,
            baseline_band[0],
            baseline_band[1],
            color=COLORS["sgd"],
            alpha=0.22,
            linewidth=0,
            zorder=1,
        )
    ax.plot(baseline_steps, baseline_mmd, color=COLORS["sgd"], label="SGD", zorder=3)

    if "natural_history_steps" in data and (
        "natural_eval_history_mean" in data or "natural_eval_histories" in data
    ):
        natural_steps = np.asarray(data["natural_history_steps"], dtype=float) + 1.0
        natural_mmd, natural_band = _gk_mmd_and_band_from_f_history(data, "natural", se_scale=se_scale)
        if natural_band is not None:
            ax.fill_between(
                natural_steps,
                natural_band[0],
                natural_band[1],
                color=COLORS["natural"],
                alpha=0.22,
                linewidth=0,
                zorder=1,
            )
        ax.plot(natural_steps, natural_mmd, color=COLORS["natural"], label="Natural SGD", zorder=3)

    if adaptive_band is not None:
        ax.fill_between(
            adaptive_steps,
            adaptive_band[0],
            adaptive_band[1],
            color=COLORS["pgd"],
            alpha=0.22,
            linewidth=0,
            zorder=1,
        )
    ax.plot(adaptive_steps, adaptive_mmd, color=COLORS["pgd"], label="PGD", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    if show_ylabel:
        ax.set_ylabel("MMD")
    ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
    if show_legend:
        ax.legend()


def _get_lhs_rhs_series(data):
    if all(key in data for key in ("last_adapt_checkpoint_steps", "last_adapt_lhs", "last_adapt_rhs")):
        step_offset = 1.0 if "theta_true" in data or "baseline_history_steps" in data else 0.0
        steps = np.asarray(data["last_adapt_checkpoint_steps"], dtype=float) + step_offset
        lhs = np.asarray(data["last_adapt_lhs"], dtype=float)
        rhs = np.asarray(data["last_adapt_rhs"], dtype=float)
        return steps, lhs, rhs

    if all(key in data for key in ("adaptive_history_steps", "adaptive_lhs_history_mean", "adaptive_rhs_history_mean")):
        steps = np.asarray(data["adaptive_history_steps"], dtype=float) + 1.0
        lhs = np.asarray(data["adaptive_lhs_history_mean"], dtype=float)
        rhs = np.asarray(data["adaptive_rhs_history_mean"], dtype=float)
        return steps, lhs, rhs

    if all(key in data for key in ("adapt_lhs_checkpoint_steps", "adapt_lhs_mean", "adapt_rhs_mean")):
        steps = np.asarray(data["adapt_lhs_checkpoint_steps"], dtype=float)
        lhs = np.asarray(data["adapt_lhs_mean"], dtype=float)
        rhs = np.asarray(data["adapt_rhs_mean"], dtype=float)
        return steps, lhs, rhs

    raise KeyError("Could not find adaptive LHS/RHS series in the provided result file.")


def _plot_theta_branch(ax, hist, cols, color, markevery):
    ax.plot(hist[:, cols[0]], hist[:, cols[1]], color=color, marker="o", markevery=markevery)
    ax.scatter(hist[0, cols[0]], hist[0, cols[1]], color="white", edgecolors=color, linewidths=1.5, s=50, zorder=4)
    ax.scatter(hist[-1, cols[0]], hist[-1, cols[1]], color=color, marker="s", s=50, zorder=5)


def _draw_gk_mean_theta_trajectories(ax_ab, ax_ck, npz_paths, show_true=True):
    method_colors = {
        "baseline": [COLORS["sgd"], COLORS["sgd_dark"]],
        "natural": [COLORS["natural"], COLORS["natural_dark"]],
        "adaptive": [COLORS["pgd"], COLORS["pgd_dark"]],
    }
    method_keys = {
        "baseline": "baseline_theta_history_mean",
        "natural": "natural_theta_history_mean",
        "adaptive": "adaptive_theta_history_mean",
    }
    theta_true = None

    for idx, npz_path in enumerate(npz_paths):
        data = _load_npz_dict(npz_path)
        theta_true = np.asarray(data["theta_true"], dtype=float)

        for method_name, history_key in method_keys.items():
            if history_key not in data:
                continue

            hist = np.asarray(data[history_key], dtype=float)
            color = method_colors[method_name][idx % len(method_colors[method_name])]
            markevery = max(1, len(hist) // 22)
            _plot_theta_branch(ax_ab, hist, (0, 1), color, markevery)
            _plot_theta_branch(ax_ck, hist, (2, 3), color, markevery)

    if show_true and theta_true is not None:
        ax_ab.scatter(theta_true[0], theta_true[1], marker="*", s=600, color=COLORS["truth"], zorder=6)
        ax_ck.scatter(theta_true[2], theta_true[3], marker="*", s=600, color=COLORS["truth"], zorder=6)

    ax_ab.set_xlabel("Parameter 1")
    ax_ab.set_ylabel("Parameter 2")
    ax_ck.set_xlabel("Parameter 3")
    ax_ck.set_ylabel("Parameter 4")
    ax_ab.grid(True, alpha=0.35)
    ax_ck.grid(True, alpha=0.35)


def make_f_vs_iteration_plot_from_gk_npz(npz_path, output_path=None, se_scale=1.96):
    data = _load_npz_dict(npz_path)
    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        _draw_gk_mmd_vs_iteration_plot(ax, data, show_legend=True, se_scale=se_scale)
        fig.tight_layout()
        return _save_figure(fig, output_path)


def make_gk_theta_trajectory_plot_from_npz(npz_path, output_path=None):
    with plt.rc_context(LOCAL_PLOT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=SUMMARY_DPI)
        _draw_gk_mean_theta_trajectories(axes[0], axes[1], [npz_path])
        fig.tight_layout()
        return _save_figure(fig, output_path)


def make_gk_summary_figure(
    mmd_npz_path,
    trajectory_npz_paths,
    lhs_rhs_npz_path,
    output_path=None,
    show_method_legend=False,
    show_lhs_rhs_legend=True,
    se_scale=1.96,
):
    mmd_data = _load_npz_dict(mmd_npz_path)
    lhs_rhs_data = _load_npz_dict(lhs_rhs_npz_path)
    lhs_rhs_steps, lhs, rhs = _get_lhs_rhs_series(lhs_rhs_data)

    with plt.rc_context(LOCAL_PLOT_RC):
        fig, axes = plt.subplots(
            1,
            4,
            figsize=SUMMARY_FIGSIZE,
            dpi=SUMMARY_DPI,
            gridspec_kw={"width_ratios": [1.15, 1.15, 1.0, 1.0]},
        )
        ax_ab, ax_ck, ax_mmd, ax_lhs_rhs = axes
        _draw_gk_mean_theta_trajectories(ax_ab, ax_ck, trajectory_npz_paths)
        _draw_gk_mmd_vs_iteration_plot(
            ax_mmd,
            mmd_data,
            show_legend=show_method_legend,
            se_scale=se_scale,
        )
        _draw_lhs_rhs_plot(ax_lhs_rhs, lhs_rhs_steps, lhs, rhs, show_legend=show_lhs_rhs_legend)

        legend_handles = [
            Line2D([0], [0], color=COLORS["sgd"], lw=LINE_WIDTH, label="SGD"),
            Line2D([0], [0], color=COLORS["natural"], lw=LINE_WIDTH, label="Natural SGD"),
            Line2D([0], [0], color=COLORS["pgd"], lw=LINE_WIDTH, label="PGD"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1))
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        return _save_figure(fig, output_path, bbox_inches="tight")


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
    return _save_figure(fig, output_path, bbox_inches="tight")

def make_lv_lengthscale_lambda_heatmap(
    results_dir,
    lambdas=(1e-4, 1e-3, 1e-2),
    metric_key="pgd_eval_mean",
    output_path=None,
    apply_log10=True,
    metric_name="MMD",  # change to "KSD" if needed
    use_sqrt_metric=False,
):
    """Render the LV lambda-lengthscale matrix in an annotated heatmap style."""
    results_dir = Path(results_dir)

    filename_pattern = re.compile(
        r"lotka_volterra_lengthscale_regularization_grid_pgd_ell_min_"
        r"(?P<ell>[^_]+)_lambda_(?P<lam>[^_]+)\.npz$"
    )

    requested_lambdas = np.asarray(lambdas, dtype=float)
    lambda_set = {float(v) for v in requested_lambdas}

    metric_by_cell = {}
    ell_values = set()

    file_pattern = (
        "lotka_volterra_lengthscale_regularization_grid_pgd_ell_min_*_lambda_*.npz"
    )

    for npz_path in sorted(results_dir.glob(file_pattern)):
        match = filename_pattern.match(npz_path.name)
        if match is None:
            continue

        ell = float(match.group("ell").replace("p", "."))
        lam = float(match.group("lam").replace("p", "."))

        if lam not in lambda_set:
            continue

        data = _load_npz_dict(npz_path)
        metric_value = float(np.asarray(data[metric_key], dtype=float))
        if use_sqrt_metric:
            metric_value = float(np.sqrt(max(metric_value, 0.0)))
        metric_by_cell[(lam, ell)] = metric_value
        ell_values.add(ell)

    if not metric_by_cell:
        raise ValueError(
            f"No matching files found in {results_dir} "
            f"for lambdas={list(requested_lambdas)}."
        )

    sorted_ells = np.asarray(sorted(ell_values), dtype=float)

    metric_grid = np.full(
        (len(requested_lambdas), len(sorted_ells)),
        np.nan,
        dtype=float,
    )

    for i, lam in enumerate(requested_lambdas):
        for j, ell in enumerate(sorted_ells):
            key = (float(lam), float(ell))
            if key in metric_by_cell:
                metric_grid[i, j] = metric_by_cell[key]

    if apply_log10:
        plot_grid = np.log10(np.maximum(metric_grid, 1e-32))
        colorbar_label = rf"$\log_{{10}}(\mathrm{{{metric_name}}})$"
    else:
        plot_grid = metric_grid
        colorbar_label = metric_name

    plot_rc = {
        **LOCAL_PLOT_RC,

        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
    }

    with plt.rc_context(plot_rc):
        fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=SUMMARY_DPI)

        im = ax.imshow(plot_grid, cmap="viridis", aspect="auto")

        ax.set_xticks(np.arange(len(sorted_ells)))
        ax.set_yticks(np.arange(len(requested_lambdas)))

        ax.set_xticklabels([f"{ell:g}" for ell in sorted_ells])
        ax.set_yticklabels([f"{lam:g}" for lam in requested_lambdas])

        ax.set_xlabel(r"$\ell_{\infty}$", fontsize=26)
        ax.set_ylabel(r"$\lambda$", fontsize=26)

        ax.grid(False)
        ax.tick_params(axis="both", labelsize=20)

        for i in range(metric_grid.shape[0]):
            for j in range(metric_grid.shape[1]):
                if np.isfinite(metric_grid[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{metric_grid[i, j]:.1e}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=11,
                    )

        cbar = fig.colorbar(im, ax=ax)

        cbar.set_label(
            colorbar_label,
            fontsize=18,
            rotation=270,
            labelpad=35,
        )

        cbar.ax.tick_params(labelsize=14)

        fig.tight_layout()

        return _save_figure(fig, output_path, bbox_inches="tight")

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    figures_dir = root / "ab"

    # make_four_panel_figure(
    #     comparison_npz=root / "results_n100f.npz",
    #     mmd_vs_n_npz_paths=[
    #         root / "results_n10f.npz",
    #         root / "results_n30f.npz",
    #         root / "results_n100f.npz",
    #         root / "results_n300f.npz",
    #     ],
    #     mmd_vs_n_ns=[10, 30, 100, 300],
    #     mmd_vs_iteration_npz=root / "results_n300f.npz",
    #     lhs_rhs_npz=root / "results_n100f.npz",
    #     output_path=figures_dir / "mmd_flow_from_utils.pdf",
    #     se_scale=1.96,
    # )
    make_lv_lengthscale_lambda_heatmap(
    figures_dir,
    lambdas=(1.0, 0.1, 0.01),
    metric_name="KSD",
)

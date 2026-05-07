import argparse
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FIGSIZE = (6, 4)
DPI = 100
SUMMARY_FIGSIZE = (30, 6.5)
SUMMARY_DPI = 150
TITLE_SIZE = 24
LABEL_SIZE = 32
TICK_SIZE = 25
LEGEND_SIZE = 32
MARKER_SIZE = 5
COMPARISON_SCATTER_SIZE = 28
LINE_WIDTH = 2.0
TITLE_PAD = 8
INLINE_LEGEND_SIZE = 18
FOUR_PANEL_LEGEND_SIZE = 32

COLORS = {
    "mmd": "#d95f02",
    "smmd": "#1b9e77",
    "svgd": "#7570b3",
    "gd": "#1b9e77",
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

METHOD_COLORS = {
    "sgd": "#006d2c",
    "adaptive_sgd": "#238b45",
    "pgd": "#d95f02",
    "fixed_pgd": "#ff9da6",
}

BOX_COLORS = [
    "#4c78a8",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ab",
]

CORRUPTION_LINESTYLES = {
    "clean": "-",
    "0": "-",
    "15": "--",
    "35": ":",
}

BASE_PLOT_RC = {
    "axes.grid": True,
    "font.family": "DejaVu Serif",
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
PLOT_RC = dict(BASE_PLOT_RC)
FOUR_PANEL_RC = {
    **LOCAL_PLOT_RC,
    "axes.labelsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 14,
    "lines.markersize": 3.5,
}
plt.rcParams.update(BASE_PLOT_RC)

NOTEBOOK_STYLE_PROFILES = {
    "mmd_flow": {
        "summary_figsize": (24, 4.8),
        "summary_dpi": 150,
        "title_size": 24,
        "label_size": 38,
        "tick_size": 38,
        "legend_size": 36,
        "marker_size": 7,
        "comparison_scatter_size": 28,
        "line_width": 3.0,
        "title_pad": 5,
        "inline_legend_size": 18,
        "four_panel_legend_size": 32,
    },
    "gnk": {
        "summary_figsize": (24, 4.8),
        "summary_dpi": 150,
        "title_size": 32,
        "label_size": 32,
        "tick_size": 25,
        "legend_size": 32,
        "marker_size": 5,
        "comparison_scatter_size": 28,
        "line_width": 2.0,
        "title_pad": 8,
        "inline_legend_size": 18,
        "four_panel_legend_size": 32,
    },
    "gnk_ablation": {
        "summary_figsize": (24, 5.0),
        "summary_dpi": 150,
        "title_size": 32,
        "label_size": 32,
        "tick_size": 26,
        "legend_size": 32,
        "marker_size": 5,
        "comparison_scatter_size": 28,
        "line_width": 2.0,
        "title_pad": 8,
        "inline_legend_size": 18,
        "four_panel_legend_size": 32,
    },
    "lv": {
        "summary_figsize": (30, 6.5),
        "summary_dpi": 150,
        "title_size": 24,
        "label_size": 40,
        "tick_size": 30,
        "legend_size": 40,
        "marker_size": 60,
        "comparison_scatter_size": 28,
        "line_width": 2.0,
        "title_pad": 8,
        "inline_legend_size": 18,
        "four_panel_legend_size": 32,
    },
    "lv_ablation": {
        "summary_figsize": (24, 5.0),
        "summary_dpi": 150,
        "title_size": 32,
        "label_size": 32,
        "tick_size": 26,
        "legend_size": 32,
        "marker_size": 5,
        "comparison_scatter_size": 28,
        "line_width": 2.0,
        "title_pad": 8,
        "inline_legend_size": 18,
        "four_panel_legend_size": 32,
    },
    "cross_method": {
        "summary_figsize": (6, 4),
        "summary_dpi": 150,
        "title_size": 28,
        "label_size": 26,
        "tick_size": 22,
        "legend_size": 20,
        "marker_size": 5,
        "comparison_scatter_size": 28,
        "line_width": 2.0,
        "title_pad": 8,
        "inline_legend_size": 18,
        "four_panel_legend_size": 20,
    },
    "combined_time": {
        "summary_figsize": (24, 5.0),
        "summary_dpi": 150,
        "title_size": 26,
        "label_size": 26,
        "tick_size": 20,
        "legend_size": 28,
        "marker_size": 5,
        "comparison_scatter_size": 28,
        "line_width": 2.0,
        "title_pad": 8,
        "inline_legend_size": 18,
        "four_panel_legend_size": 28,
    },
}


def _set_plot_style(profile_name):
    global SUMMARY_FIGSIZE, SUMMARY_DPI, TITLE_SIZE, LABEL_SIZE, TICK_SIZE
    global LEGEND_SIZE, MARKER_SIZE, COMPARISON_SCATTER_SIZE, LINE_WIDTH
    global TITLE_PAD, INLINE_LEGEND_SIZE, FOUR_PANEL_LEGEND_SIZE
    global BASE_PLOT_RC, LOCAL_PLOT_RC, PLOT_RC, FOUR_PANEL_RC

    profile = NOTEBOOK_STYLE_PROFILES[profile_name]
    SUMMARY_FIGSIZE = profile["summary_figsize"]
    SUMMARY_DPI = profile["summary_dpi"]
    TITLE_SIZE = profile["title_size"]
    LABEL_SIZE = profile["label_size"]
    TICK_SIZE = profile["tick_size"]
    LEGEND_SIZE = profile["legend_size"]
    MARKER_SIZE = profile["marker_size"]
    COMPARISON_SCATTER_SIZE = profile["comparison_scatter_size"]
    LINE_WIDTH = profile["line_width"]
    TITLE_PAD = profile["title_pad"]
    INLINE_LEGEND_SIZE = profile["inline_legend_size"]
    FOUR_PANEL_LEGEND_SIZE = profile["four_panel_legend_size"]

    BASE_PLOT_RC = {
        "axes.grid": True,
        "font.family": "DejaVu Serif",
        "font.serif": ["Times New Roman"],
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath, amsfonts, mathrsfs, amssymb}",
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
    PLOT_RC = dict(BASE_PLOT_RC)
    FOUR_PANEL_RC = {
        **LOCAL_PLOT_RC,
        "axes.labelsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 14,
        "lines.markersize": 3.5,
    }
    plt.rcParams.update(BASE_PLOT_RC)


def set_plot_style(profile_name):
    _set_plot_style(profile_name)
    return dict(LOCAL_PLOT_RC)


def get_plot_style(profile_name=None):
    if profile_name is not None:
        return set_plot_style(profile_name)
    return dict(LOCAL_PLOT_RC)


# Shared helpers
def _load_npz_dict(npz_path):
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def load_npz_dict(npz_path):
    return _load_npz_dict(npz_path)


def _save_figure(fig, output_path=None, **savefig_kwargs):
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, **savefig_kwargs)
    return fig


def save_figure(fig, output_path=None, **savefig_kwargs):
    return _save_figure(fig, output_path=output_path, **savefig_kwargs)


def _make_single_color_boxplot(ax, series, labels, color, xlabel, ylabel, title=None, logy=True):
    bp = ax.boxplot(series, patch_artist=True, widths=0.6)
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(BOX_COLORS[idx % len(BOX_COLORS)])
        patch.set_alpha(0.8)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2.0)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, pad=TITLE_PAD)
    if logy:
        ax.set_yscale("log")
    return bp


def make_single_color_boxplot(ax, series, labels, color, xlabel, ylabel, title=None, logy=True):
    return _make_single_color_boxplot(
        ax,
        series=series,
        labels=labels,
        color=color,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        logy=logy,
    )


def _parse_heatmap_float(value):
    return float(value.replace("p", "."))


def _draw_theta_error_heatmap(error_grid, ell_values, lambda_values, output_path=None):
    masked_grid = np.ma.masked_invalid(error_grid)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")

    figure_height = 5.0 if len(lambda_values) > 1 else 2.8
    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=(8.0, figure_height), dpi=SUMMARY_DPI)
        im = ax.imshow(masked_grid, cmap=cmap, aspect="auto")

        ax.set_xticks(np.arange(len(ell_values)))
        ax.set_yticks(np.arange(len(lambda_values)))
        ax.set_xticklabels([f"{ell:g}" for ell in ell_values])
        ax.set_yticklabels([f"{lam:g}" for lam in lambda_values])
        ax.set_xlabel(r"$\ell_{\infty}$")
        ax.set_ylabel(r"$\lambda$")
        ax.grid(False)

        for i in range(error_grid.shape[0]):
            for j in range(error_grid.shape[1]):
                if np.isfinite(error_grid[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{error_grid[i, j]:.2e}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=12,
                    )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$\|\bar{\theta}_{final} - \theta_{true}\|_2$", rotation=270, labelpad=35)
        fig.tight_layout()
        return _save_figure(fig, output_path, bbox_inches="tight")


def _draw_lengthscale_pair_heatmaps(
    grids,
    x_values,
    y_values,
    lambda_values,
    colorbar_label,
    output_path=None,
):
    masked_grids = [np.ma.masked_invalid(np.asarray(grid, dtype=float)) for grid in grids]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")

    finite_values = np.concatenate(
        [grid.compressed() for grid in masked_grids if np.ma.count(grid) > 0]
    )
    vmin = float(np.min(finite_values)) if finite_values.size else None
    vmax = float(np.max(finite_values)) if finite_values.size else None

    ncols = len(lambda_values)
    with plt.rc_context(LOCAL_PLOT_RC):
        fig, axes = plt.subplots(
            1,
            ncols,
            figsize=(5.2 * ncols, 4.8),
            dpi=SUMMARY_DPI,
            squeeze=False,
        )
        axes = axes[0]
        ims = []

        for ax, lam, grid in zip(axes, lambda_values, masked_grids):
            im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
            ims.append(im)
            ax.set_xticks(np.arange(len(x_values)))
            ax.set_yticks(np.arange(len(y_values)))
            ax.set_xticklabels([f"{value:g}" for value in x_values])
            ax.set_yticklabels([f"{value:g}" for value in y_values])
            ax.set_xlabel(r"$\ell_0$")
            ax.set_ylabel(r"$\ell_{\infty}$")
            ax.set_title(rf"$\lambda = {lam:g}$", pad=TITLE_PAD)
            ax.grid(False)

            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if np.isfinite(grid[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{grid[i, j]:.2e}",
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=10,
                        )

        cbar = fig.colorbar(ims[-1], ax=axes, fraction=0.03, pad=0.03)
        cbar.set_label(colorbar_label, rotation=270, labelpad=30)
        fig.tight_layout()
        return _save_figure(fig, output_path, bbox_inches="tight")


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


# MMD flow plotting helpers
def _sample_target_mog(num_samples, seed=0, radius=2.0, std=0.2, k=8):
    rng = np.random.default_rng(seed)
    angles = 2.0 * np.pi * np.arange(k, dtype=np.float64) / k
    means = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles)],
        axis=1,
    )
    component_ids = rng.integers(0, k, size=num_samples)
    return means[component_ids] + rng.normal(scale=std, size=(num_samples, 2))


def _sample_initial_particles(num_particles, seed=0, d=2):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(num_particles, d)).astype(np.float64)


def _draw_initial_particles_plot(
    ax,
    initial_samples,
    xlim=None,
    ylim=None,
    show_legend=True,
):
    initial_samples = np.asarray(initial_samples, dtype=float)
    ax.scatter(
        initial_samples[:, 0] / 3.0,
        initial_samples[:, 1] / 3.0,
        s=COMPARISON_SCATTER_SIZE,
        facecolors="none",
        edgecolors="#444444",
        linewidths=0.9,
        alpha=0.9,
        label="Initial",
    )
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.15, linewidth=0.8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if show_legend:
        ax.legend(loc="lower left", fontsize=INLINE_LEGEND_SIZE, frameon=True, facecolor="white", framealpha=0.9)


def _comparison_plot_limits(target_samples, fixed_samples, adaptive_samples, initial_samples=None):
    all_samples = [target_samples, fixed_samples, adaptive_samples]
    if initial_samples is not None:
        all_samples.append(np.asarray(initial_samples, dtype=float))
    all_samples = np.vstack(all_samples)
    mins = all_samples.min(axis=0)
    maxs = all_samples.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = np.max(maxs - mins)
    half_width = 0.55 * span
    x_min, x_max = center[0] - half_width, center[0] + half_width
    y_min, y_max = center[1] - half_width, center[1] + half_width

    return x_min, x_max, y_min, y_max


def _draw_comparison_plot(
    ax,
    target_samples,
    fixed_samples,
    adaptive_samples,
    initial_samples=None,
    show_legend=True,
):
    x_min, x_max, y_min, y_max = _comparison_plot_limits(
        target_samples,
        fixed_samples,
        adaptive_samples,
        initial_samples=initial_samples,
    )
    ax.scatter(
        fixed_samples[:, 0],
        fixed_samples[:, 1],
        s=COMPARISON_SCATTER_SIZE,
        c=COLORS["mmd"],
        alpha=0.55,
        edgecolors="none",
        label="Fixed",
    )
    ax.scatter(
        adaptive_samples[:, 0],
        adaptive_samples[:, 1],
        s=COMPARISON_SCATTER_SIZE,
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
        ax.legend(loc="lower left", fontsize=INLINE_LEGEND_SIZE, frameon=True, facecolor="white", framealpha=0.9)


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
    lower = np.maximum(mmd - se, 0.0)
    upper = mmd + se
    return mmd, (lower, upper)


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
    ax.set_xlabel(r"Particle Number $N$")
    if show_ylabel:
        ax.set_ylabel(r"$\mathrm{MMD}_{\ell_\infty}(\mathbb{P}, \mathbb{Q})$", labelpad=4)
    ax.grid(True, which="major", alpha=0.15, linewidth=0.8)
    if show_legend:
        ax.legend(loc="lower left", fontsize=INLINE_LEGEND_SIZE, frameon=True, facecolor="white", framealpha=0.9)


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
        ax.fill_between(adaptive_steps, adaptive_band[0], adaptive_band[1], color=COLORS["smmd"], alpha=0.22, linewidth=0)
    ax.plot(fixed_steps, fixed_mmd, color=COLORS["mmd"], label="Fixed", zorder=3)
    ax.plot(adaptive_steps, adaptive_mmd, color=COLORS["smmd"], label="Adaptive", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    if show_ylabel:
        ax.set_ylabel(r"$\mathrm{MMD}_{\ell_\infty}(\mathbb{P}, \mathbb{Q})$", labelpad=4)
    ax.set_ylim(1e-2, 0.3)
    # ax.set_title("MMD vs Iteration", pad=TITLE_PAD)
    ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
    if show_legend:
        ax.legend(loc="best", fontsize=INLINE_LEGEND_SIZE, frameon=True, facecolor="white", framealpha=0.9)


def _draw_lhs_rhs_plot(ax, steps, lhs, rhs, show_legend=True, show_ylabel=True):
    steps = np.asarray(steps, dtype=float)
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    ax.plot(steps, lhs, color="#D62728", label="LHS")
    ax.plot(steps, rhs, color="#111111", label="RHS")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    # ax.set_title("LHS vs RHS", pad=TITLE_PAD)
    ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
    if show_legend:
        ax.legend(loc="lower left", fontsize=INLINE_LEGEND_SIZE * 1.5, frameon=True, facecolor="white", framealpha=0.9)


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


def make_mmd_vs_n_plot(ns, fixed_mmd, adaptive_mmd, fixed_band=None, adaptive_band=None, output_path=None):
    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        _draw_mmd_vs_n_plot(ax, ns, fixed_mmd, adaptive_mmd, fixed_band=fixed_band, adaptive_band=adaptive_band)
        fig.tight_layout()
        return _save_figure(fig, output_path)


def make_mmd_vs_iteration_plot(
    fixed_mmd,
    adaptive_mmd,
    fixed_steps=None,
    adaptive_steps=None,
    fixed_band=None,
    adaptive_band=None,
    output_path=None,
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
        fixed_steps=fixed_steps,
        adaptive_steps=adaptive_steps,
        fixed_band=fixed_band,
        adaptive_band=adaptive_band,
        output_path=output_path,
    )


def make_lhs_rhs_plot_from_npz(npz_path, output_path=None):
    data = _load_npz_dict(npz_path)
    steps = np.asarray(data["last_adapt_checkpoint_steps"], dtype=float)
    lhs = np.asarray(data["last_adapt_lhs"], dtype=float)
    rhs = np.asarray(data["last_adapt_rhs"], dtype=float)
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
        fixed_band=fixed_band,
        adaptive_band=adaptive_band,
        output_path=output_path,
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
    comparison_seed = int(np.asarray(comparison_data["fixed_finals"]).shape[0] - 1)
    initial_samples = _sample_initial_particles(fixed_samples.shape[0], seed=comparison_seed, d=fixed_samples.shape[1])

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
    lhs_rhs_steps = np.asarray(lhs_rhs_data["last_adapt_checkpoint_steps"], dtype=float)
    lhs_values = np.asarray(lhs_rhs_data["last_adapt_lhs"], dtype=float)
    rhs_values = np.asarray(lhs_rhs_data["last_adapt_rhs"], dtype=float)
    with plt.rc_context(FOUR_PANEL_RC):
        fig = plt.figure(figsize=(25 / 1.2, 5), dpi=100, constrained_layout=True)
        gs = fig.add_gridspec(
            2,
            4,
            width_ratios=[1.0, 1.0, 1.0, 1.0],
            height_ratios=[1.0, 1.0],
            wspace=0.05,
            hspace=-0.05,
        )
        ax_initial = fig.add_subplot(gs[:, 0])
        ax_comparison = fig.add_subplot(gs[:, 1])
        ax_mmd_iter = fig.add_subplot(gs[0, 2])
        ax_mmd_n = fig.add_subplot(gs[1, 2])
        ax_lhs_rhs = fig.add_subplot(gs[:, 3])

        comparison_xlim, comparison_ylim = _comparison_plot_limits(
            target_samples,
            fixed_samples,
            adaptive_samples,
        )[:2], _comparison_plot_limits(
            target_samples,
            fixed_samples,
            adaptive_samples,
        )[2:]

        _draw_initial_particles_plot(
            ax_initial,
            initial_samples,
            xlim=comparison_xlim,
            ylim=comparison_ylim,
            show_legend=True,
        )
        _draw_comparison_plot(
            ax_comparison,
            target_samples,
            fixed_samples,
            adaptive_samples,
            show_legend=True,
        )
        _draw_mmd_vs_n_plot(
            ax_mmd_n,
            mmd_vs_n_ns,
            mmd_vs_n_fixed,
            mmd_vs_n_adaptive,
            fixed_band=mmd_vs_n_fixed_band,
            adaptive_band=mmd_vs_n_adaptive_band,
            show_legend=True,
        )
        _draw_mmd_vs_iteration_plot(
            ax_mmd_iter,
            iteration_fixed_mmd,
            iteration_adaptive_mmd,
            fixed_steps=iteration_fixed_steps,
            adaptive_steps=iteration_adaptive_steps,
            fixed_band=iteration_fixed_band,
            adaptive_band=iteration_adaptive_band,
            show_legend=True,
        )
        _draw_lhs_rhs_plot(
            ax_lhs_rhs,
            lhs_rhs_steps,
            lhs_values,
            rhs_values,
        )
        return _save_figure(fig, output_path, bbox_inches="tight")


# G-and-K plotting helpers
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
    ax.plot(baseline_steps, baseline_mmd, color=COLORS["sgd"], label="GD", zorder=3)
    if "natural_history_steps" in data and "natural_eval_history_mean" in data:
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
    ax.plot(adaptive_steps, adaptive_mmd, color=COLORS["pgd"], label="PGD (ours)", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    if show_ylabel:
        ax.set_ylabel(r"$\mathrm{MMD}_{\ell_\infty}(\mathbb{P}_\theta, \mathbb{Q})$")
    ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
    if show_legend:
        ax.legend()


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
            Line2D([0], [0], color=COLORS["sgd"], lw=LINE_WIDTH, label="GD"),
            Line2D([0], [0], color=COLORS["natural"], lw=LINE_WIDTH, label="PGD (Briol et al.)"),
            Line2D([0], [0], color=COLORS["pgd"], lw=LINE_WIDTH, label="PGD (ours)"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1))
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        return _save_figure(fig, output_path, bbox_inches="tight")


def make_gk_mmd_vs_time_plot(npz_paths, output_path=None):
    method_specs = [
        ("baseline", "GD", COLORS["sgd"]),
        ("natural", "PGD (Briol et al.)", COLORS["natural"]),
        ("adaptive", "PGD (ours)", COLORS["pgd"]),
    ]

    if isinstance(npz_paths, (str, Path)):
        data = _load_npz_dict(npz_paths)
        series = []
        for prefix, label, color in method_specs:
            required = [
                f"{prefix}_checkpoint_iterations",
                f"{prefix}_checkpoint_elapsed_mean",
                f"{prefix}_checkpoint_eval_mean",
            ]
            missing = [key for key in required if key not in data]
            if missing:
                raise KeyError(
                    f"{npz_paths} is missing G-and-K checkpoint data for '{prefix}'. "
                    "Regenerate this file with the updated g_and_k.py checkpoint saver."
                )
            times = np.asarray(data[f"{prefix}_checkpoint_elapsed_mean"], dtype=float)
            mmds = np.sqrt(np.asarray(data[f"{prefix}_checkpoint_eval_mean"], dtype=float))
            order = np.argsort(times)
            series.append((times[order], mmds[order], label, color))
    else:
        series = []
        for prefix, label, color in method_specs:
            times = []
            mmds = []

            for npz_path in npz_paths:
                data = _load_npz_dict(npz_path)
                times.append(float(np.asarray(data[f"{prefix}_elapsed_mean"], dtype=float)))
                mmds.append(np.sqrt(float(np.asarray(data[f"{prefix}_eval_mean"], dtype=float))))

            times = np.asarray(times, dtype=float)
            mmds = np.asarray(mmds, dtype=float)
            order = np.argsort(times)
            series.append((times[order], mmds[order], label, color))

    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=(10.5, 7.6), dpi=150)

        for times, mmds, label, color in series:
            ax.plot(
                times,
                mmds,
                color=color,
                marker="o",
                linewidth=2.8,
                markersize=8,
                label=label,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\mathrm{MMD}_{\ell_\infty}(\mathbb{P}_\theta, \mathbb{Q})$")
        ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        return _save_figure(fig, output_path, bbox_inches="tight")


def make_gk_theta_error_heatmap(results_dir, output_path):
    results_dir = Path(results_dir)
    dirname_pattern = re.compile(
        r"g_and_k_lengthscale_regularization_grid_ell_min_"
        r"(?P<ell>[^_]+)_lambda_(?P<lam>[^/]+)$"
    )

    error_by_cell = {}
    ell_values = set()
    lambda_values = set()

    for npz_path in sorted(results_dir.rglob("*.npz")):
        relative_parent = npz_path.parent.relative_to(results_dir).as_posix()
        match = dirname_pattern.match(relative_parent)
        if match is None:
            continue

        ell = _parse_heatmap_float(match.group("ell"))
        lam = _parse_heatmap_float(match.group("lam"))
        data = _load_npz_dict(npz_path)
        theta_true = np.asarray(data["theta_true"], dtype=float)
        theta_final = np.asarray(data["adaptive_theta_mean"], dtype=float)
        theta_error = float(np.linalg.norm(theta_final - theta_true))

        error_by_cell[(lam, ell)] = theta_error
        ell_values.add(ell)
        lambda_values.add(lam)

    if not error_by_cell:
        raise ValueError(f"No G-and-K heatmap files found in {results_dir}")

    sorted_ells = np.asarray(sorted(ell_values), dtype=float)
    sorted_lambdas = np.asarray(sorted(lambda_values), dtype=float)
    error_grid = np.full((len(sorted_lambdas), len(sorted_ells)), np.nan, dtype=float)

    for i, lam in enumerate(sorted_lambdas):
        for j, ell in enumerate(sorted_ells):
            if (float(lam), float(ell)) in error_by_cell:
                error_grid[i, j] = error_by_cell[(float(lam), float(ell))]

    return _draw_theta_error_heatmap(error_grid, sorted_ells, sorted_lambdas, output_path)


# Lotka-Volterra plotting helpers
def _lv_plot_branch(ax, hist, color, markevery, start_marker_color=None, start_marker="o", start_marker_size=220):
    ax.plot(hist[:, 0], hist[:, 1], color=color, marker="o", markevery=markevery, markersize=5.0)
    if start_marker_color is None:
        start_marker_color = "white"
    ax.scatter(
        hist[0, 0],
        hist[0, 1],
        color=start_marker_color,
        marker=start_marker,
        edgecolors="none",
        linewidths=0.0,
        s=start_marker_size,
        zorder=5,
    )
    ax.scatter(hist[-1, 0], hist[-1, 1], color=color, marker="s", s=48, zorder=5)


def _lv_draw_theta12_trajectory(ax, npz_paths, show_true=True, init_markers=None, init_colors=None):
    path_colors = [
        (COLORS["gd"], COLORS["natural"], COLORS["pgd"]),
        (COLORS["sgd_dark"], COLORS["natural_dark"], COLORS["pgd_dark"]),
    ]
    if init_markers is None:
        init_markers = ["o", "^", "D", "v"]
    if init_colors is None:
        init_colors = ["#7B3294", "#4D4D4D", "#7B3294", "#4D4D4D"]

    theta_true = None
    for idx, npz_path in enumerate(npz_paths):
        data = _load_npz_dict(npz_path)
        sgd = np.asarray(data["sgd_theta_history_mean"], dtype=float)[:, :2]
        natural = np.asarray(data["natural_theta_history_mean"], dtype=float)[:, :2] if "natural_theta_history_mean" in data else None
        pgd = np.asarray(data["pgd_theta_history_mean"], dtype=float)[:, :2]
        theta_true = np.asarray(data["theta_true"], dtype=float)
        sgd_color, natural_color, pgd_color = path_colors[min(idx, len(path_colors) - 1)]
        init_marker = init_markers[idx % len(init_markers)]
        init_color = init_colors[idx % len(init_colors)]
        _lv_plot_branch(ax, sgd, sgd_color, max(1, len(sgd) // 22), start_marker_color=init_color, start_marker=init_marker)
        if natural is not None:
            _lv_plot_branch(ax, natural, natural_color, max(1, len(natural) // 22), start_marker_color=init_color, start_marker=init_marker)
        _lv_plot_branch(ax, pgd, pgd_color, max(1, len(pgd) // 22), start_marker_color=init_color, start_marker=init_marker)

    if show_true and theta_true is not None:
        ax.scatter(theta_true[0], theta_true[1], marker="*", s=900, color=COLORS["truth"], zorder=6)

    ax.set_xlabel(r"Parameter 1 ($\theta_1$)")
    ax.set_ylabel(r"Parameter 2 ($\theta_2$)")
    ax.grid(True, alpha=0.24, linewidth=0.8)


def make_lv_theta_trajectory_plot(npz_paths, output_path=None, show_true=True):
    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=(5.2, 4.9), dpi=150)
        _lv_draw_theta12_trajectory(ax, npz_paths, show_true=show_true)
        fig.tight_layout()
        return _save_figure(fig, output_path)


def _lv_draw_history_panel(
    ax,
    bundles,
    param_idx,
    ylabel,
    true_theta,
    show_xlabel=True,
    se_scale=1.96,
    true_only=True,
    init_markers=None,
    init_colors=None,
):
    for idx, (data, color_sgd, color_natural, color_pgd, ls_sgd, ls_natural, ls_pgd) in enumerate(bundles):
        sgd_steps = np.asarray(data["sgd_history_steps"], dtype=float) + 1.0
        pgd_steps = np.asarray(data["pgd_history_steps"], dtype=float) + 1.0
        sgd_hist = np.asarray(data["sgd_theta_histories"], dtype=float)[:, :, param_idx]
        pgd_hist = np.asarray(data["pgd_theta_histories"], dtype=float)[:, :, param_idx]
        natural_hist = np.asarray(data["natural_theta_histories"], dtype=float)[:, :, param_idx] if "natural_theta_histories" in data else None
        natural_steps = np.asarray(data["natural_history_steps"], dtype=float) + 1.0 if "natural_history_steps" in data else None

        sgd_mean = sgd_hist.mean(axis=0)
        pgd_mean = pgd_hist.mean(axis=0)
        sgd_se = sgd_hist.std(axis=0, ddof=0) / np.sqrt(max(sgd_hist.shape[0], 1))
        pgd_se = pgd_hist.std(axis=0, ddof=0) / np.sqrt(max(pgd_hist.shape[0], 1))
        init_marker = None if init_markers is None else init_markers[idx % len(init_markers)]
        init_color = None if init_colors is None else init_colors[idx % len(init_colors)]

        ax.plot(
            sgd_steps,
            sgd_mean,
            color=color_sgd,
            linestyle=ls_sgd,
            linewidth=2.0,
            marker=None,
        )
        ax.fill_between(
            sgd_steps,
            sgd_mean - se_scale * sgd_se,
            sgd_mean + se_scale * sgd_se,
            color=color_sgd,
            alpha=0.12,
        )
        if natural_hist is not None:
            natural_mean = natural_hist.mean(axis=0)
            natural_se = natural_hist.std(axis=0, ddof=0) / np.sqrt(max(natural_hist.shape[0], 1))
            ax.plot(
                natural_steps,
                natural_mean,
                color=color_natural,
                linestyle=ls_natural,
                linewidth=2.0,
                marker=None,
            )
            ax.fill_between(
                natural_steps,
                natural_mean - se_scale * natural_se,
                natural_mean + se_scale * natural_se,
                color=color_natural,
                alpha=0.12,
            )
        ax.plot(
            pgd_steps,
            pgd_mean,
            color=color_pgd,
            linestyle=ls_pgd,
            linewidth=2.0,
            marker=None,
        )
        ax.fill_between(
            pgd_steps,
            pgd_mean - se_scale * pgd_se,
            pgd_mean + se_scale * pgd_se,
            color=color_pgd,
            alpha=0.12,
        )
        if init_marker is not None:
            marker_color = init_color if init_color is not None else "white"
            ax.scatter(
                sgd_steps[0],
                sgd_mean[0],
                color=marker_color,
                marker=init_marker,
                edgecolors="none",
                linewidths=0.0,
                s=220,
                zorder=5,
            )
            if natural_hist is not None:
                ax.scatter(
                    natural_steps[0],
                    natural_mean[0],
                    color=marker_color,
                    marker=init_marker,
                    edgecolors="none",
                    linewidths=0.0,
                    s=220,
                    zorder=5,
                )
            ax.scatter(
                pgd_steps[0],
                pgd_mean[0],
                color=marker_color,
                marker=init_marker,
                edgecolors="none",
                linewidths=0.0,
                s=220,
                zorder=5,
            )

    if true_only:
        ax.axhline(float(true_theta[param_idx]), color="0.35", linestyle="--", linewidth=1.2)

    ax.set_xscale("log")
    ax.set_xlabel("Iteration" if show_xlabel else "")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.28, linewidth=0.8)


def make_lv_param_history_plot_from_npz(
    npz_path,
    param_idx,
    output_path=None,
    se_scale=1.96,
    solid_for_both=True,
    true_only=True,
    show_xlabel=True,
):
    data = _load_npz_dict(npz_path)
    bundles = [
        (
            data,
            COLORS["gd"],
            COLORS["pgd"],
            "-" if solid_for_both else "--",
            "-",
        )
    ]
    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=(7.1, 4.8), dpi=150)
        _lv_draw_history_panel(
            ax,
            bundles,
            param_idx=param_idx,
            ylabel=rf"$\theta_{param_idx + 1}$",
            true_theta=np.asarray(data["theta_true"], dtype=float),
            show_xlabel=show_xlabel,
            se_scale=se_scale,
            true_only=true_only,
        )
        fig.tight_layout()
        return _save_figure(fig, output_path)


def make_lv_dual_init_history_plot(
    npz_paths,
    param_idx,
    output_path=None,
    figsize=(6, 2),
    dpi=100,
    se_scale=1.96,
):
    bundles = []
    for idx, npz_path in enumerate(npz_paths):
        data = _load_npz_dict(npz_path)
        if idx == 0:
            bundles.append((data, COLORS["gd"], COLORS["pgd"], "-", "-"))
        else:
            bundles.append((data, COLORS["gd"], COLORS["pgd"], "--", "--"))

    true_theta = np.asarray(bundles[0][0]["theta_true"], dtype=float)
    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        _lv_draw_history_panel(
            ax,
            bundles,
            param_idx=param_idx,
            ylabel=rf"$\theta_{param_idx + 1}$",
            true_theta=true_theta,
            show_xlabel=True,
            se_scale=se_scale,
            true_only=True,
        )
        fig.tight_layout()
        return _save_figure(fig, output_path)


def make_lv_corruption_history_plot(
    npz_paths,
    param_idx,
    output_path=None,
    figsize=(6, 2),
    dpi=100,
    se_scale=1.96,
):
    bundles = []
    for idx, npz_path in enumerate(npz_paths):
        data = _load_npz_dict(npz_path)
        corruption_key = "15" if idx == 0 else "35"
        ls = CORRUPTION_LINESTYLES[corruption_key]
        bundles.append((data, COLORS["gd"], COLORS["pgd"], ls, ls))

    true_theta = np.asarray(bundles[0][0]["theta_true"], dtype=float)
    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        _lv_draw_history_panel(
            ax,
            bundles,
            param_idx=param_idx,
            ylabel=rf"$\theta_{param_idx + 1}$",
            true_theta=true_theta,
            show_xlabel=True,
            se_scale=se_scale,
            true_only=True,
        )
        fig.tight_layout()
        return _save_figure(fig, output_path)


def make_lv_five_panel_summary(
    clean_npz_paths,
    corruption_npz_paths,
    output_path=None,
    figsize=(24, 5),
    dpi=100,
    se_scale=1.96,
):
    clean_data = [_load_npz_dict(path) for path in clean_npz_paths]
    cor_data = [_load_npz_dict(path) for path in corruption_npz_paths]
    theta_true = np.asarray(clean_data[0]["theta_true"], dtype=float)
    init_markers = ["o", "^"]
    init_colors = ["#7B3294", "#4D4D4D"]

    with plt.rc_context(LOCAL_PLOT_RC):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.05, 1.05], wspace=0.24, hspace=0.20)
        ax_traj = fig.add_subplot(gs[:, 0])
        ax_clean_1 = fig.add_subplot(gs[0, 1])
        ax_clean_2 = fig.add_subplot(gs[1, 1])
        ax_cor_1 = fig.add_subplot(gs[0, 2])
        ax_cor_2 = fig.add_subplot(gs[1, 2])

        _lv_draw_theta12_trajectory(ax_traj, clean_npz_paths, show_true=True, init_markers=init_markers, init_colors=init_colors)

        clean_bundles = [
            (clean_data[0], COLORS["gd"], COLORS["natural"], COLORS["pgd"], CORRUPTION_LINESTYLES["0"], CORRUPTION_LINESTYLES["0"], CORRUPTION_LINESTYLES["0"]),
            (clean_data[1], COLORS["gd"], COLORS["natural"], COLORS["pgd"], CORRUPTION_LINESTYLES["0"], CORRUPTION_LINESTYLES["0"], CORRUPTION_LINESTYLES["0"]),
        ]
        cor_bundles = [
            (cor_data[0], COLORS["gd"], COLORS["natural"], COLORS["pgd"], CORRUPTION_LINESTYLES["15"], CORRUPTION_LINESTYLES["15"], CORRUPTION_LINESTYLES["15"]),
            (cor_data[1], COLORS["gd"], COLORS["natural"], COLORS["pgd"], CORRUPTION_LINESTYLES["35"], CORRUPTION_LINESTYLES["35"], CORRUPTION_LINESTYLES["35"]),
        ]

        _lv_draw_history_panel(ax_clean_1, clean_bundles, 0, r"$\theta_1$", theta_true, show_xlabel=False, se_scale=se_scale, init_markers=init_markers, init_colors=init_colors)
        _lv_draw_history_panel(ax_clean_2, clean_bundles, 1, r"$\theta_2$", theta_true, show_xlabel=True, se_scale=se_scale, init_markers=init_markers, init_colors=init_colors)
        _lv_draw_history_panel(ax_cor_1, cor_bundles, 0, r"$\theta_1$", theta_true, show_xlabel=False, se_scale=se_scale)
        _lv_draw_history_panel(ax_cor_2, cor_bundles, 1, r"$\theta_2$", theta_true, show_xlabel=True, se_scale=se_scale)
        ax_clean_1.set_xticks([])
        ax_cor_1.set_xticks([])

        left_handles = [
            Line2D([0], [0], color=COLORS["gd"], linestyle="-", linewidth=2.2, label="GD"),
            Line2D([0], [0], color=COLORS["natural"], linestyle="-", linewidth=2.2, label="PGD (Briol et al.)"),
            Line2D([0], [0], color=COLORS["pgd"], linestyle="-", linewidth=2.2, label="PGD"),
        ]
        right_handles = [
            Line2D([0], [0], color="black", linestyle=CORRUPTION_LINESTYLES["0"], linewidth=2.2, label="0\%"),
            Line2D([0], [0], color="black", linestyle=CORRUPTION_LINESTYLES["15"], linewidth=2.2, label="15\%"),
            Line2D([0], [0], color="black", linestyle=CORRUPTION_LINESTYLES["35"], linewidth=2.2, label="35\%"),
        ]

        fig.legend(
            left_handles,
            [h.get_label() for h in left_handles],
            loc="upper left",
            bbox_to_anchor=(0.09, 1.06),
            ncol=3,
            frameon=False,
            fontsize=40,
            handlelength=2.2,
            columnspacing=1.4,
            handletextpad=0.6,
        )
        fig.legend(
            right_handles,
            [h.get_label() for h in right_handles],
            loc="upper right",
            bbox_to_anchor=(0.92, 1.06),
            ncol=3,
            frameon=False,
            fontsize=40,
            handlelength=2.2,
            columnspacing=1.2,
            handletextpad=0.6,
        )

        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return _save_figure(fig, output_path, bbox_inches="tight")


def make_lv_theta_vs_time_budget_plot(path_map, output_path=None, se_scale=1.96, init_time_factor=0.85):
    init_styles = {
        "50, 60": {"marker": "o", "init_color": "#7B3294"},
        "90, 90": {"marker": "^", "init_color": "#4D4D4D"},
    }

    with plt.rc_context(LOCAL_PLOT_RC):
        fig, axes = plt.subplots(2, 1, figsize=(6.3, 5.4), dpi=150, sharex=True)
        theta_true = None
        global_max_time = 0.0
        left_limit = np.inf

        for init_label, paths in path_map.items():
            style = init_styles.get(init_label, {"marker": "o", "init_color": "#4D4D4D"})
            sgd_time = []
            pgd_time = []
            sgd_theta = []
            pgd_theta = []
            theta0 = None
            sgd_theta_se = []
            pgd_theta_se = []

            for npz_path in paths:
                data = _load_npz_dict(npz_path)
                theta_true = np.asarray(data["theta_true"], dtype=float)
                theta0 = np.asarray(data["theta0"], dtype=float)
                sgd_time.append(float(np.asarray(data["sgd_elapsed_mean"], dtype=float)))
                pgd_time.append(float(np.asarray(data["pgd_elapsed_mean"], dtype=float)))
                sgd_theta.append(np.asarray(data["sgd_theta_mean"], dtype=float)[:2])
                pgd_theta.append(np.asarray(data["pgd_theta_mean"], dtype=float)[:2])
                sgd_theta_se.append(np.asarray(data["sgd_theta_se"], dtype=float)[:2])
                pgd_theta_se.append(np.asarray(data["pgd_theta_se"], dtype=float)[:2])

            sgd_time = np.asarray(sgd_time, dtype=float)
            pgd_time = np.asarray(pgd_time, dtype=float)
            sgd_theta = np.asarray(sgd_theta, dtype=float)
            pgd_theta = np.asarray(pgd_theta, dtype=float)
            sgd_theta_se = np.asarray(sgd_theta_se, dtype=float)
            pgd_theta_se = np.asarray(pgd_theta_se, dtype=float)

            global_max_time = max(global_max_time, float(np.max(sgd_time)), float(np.max(pgd_time)))
            sgd_order = np.argsort(sgd_time)
            pgd_order = np.argsort(pgd_time)
            init_time = init_time_factor * min(float(np.min(sgd_time)), float(np.min(pgd_time)))
            left_limit = min(left_limit, 0.8 * init_time)

            for param_idx, ax in enumerate(axes):
                sgd_plot_time = np.concatenate(([init_time], sgd_time[sgd_order]))
                sgd_plot_theta = np.concatenate(([theta0[param_idx]], sgd_theta[sgd_order, param_idx]))
                pgd_plot_time = np.concatenate(([init_time], pgd_time[pgd_order]))
                pgd_plot_theta = np.concatenate(([theta0[param_idx]], pgd_theta[pgd_order, param_idx]))

                ax.plot(
                    sgd_plot_time,
                    sgd_plot_theta,
                    color=COLORS["gd"],
                    linewidth=2.0,
                    marker=None,
                )
                ax.fill_between(
                    sgd_time[sgd_order],
                    sgd_theta[sgd_order, param_idx] - se_scale * sgd_theta_se[sgd_order, param_idx],
                    sgd_theta[sgd_order, param_idx] + se_scale * sgd_theta_se[sgd_order, param_idx],
                    color=COLORS["gd"],
                    alpha=0.12,
                )
                ax.plot(
                    pgd_plot_time,
                    pgd_plot_theta,
                    color=COLORS["pgd"],
                    linewidth=2.0,
                    marker=None,
                )
                ax.fill_between(
                    pgd_time[pgd_order],
                    pgd_theta[pgd_order, param_idx] - se_scale * pgd_theta_se[pgd_order, param_idx],
                    pgd_theta[pgd_order, param_idx] + se_scale * pgd_theta_se[pgd_order, param_idx],
                    color=COLORS["pgd"],
                    alpha=0.12,
                )
                ax.scatter(
                    init_time,
                    theta0[param_idx],
                    color=style["init_color"],
                    marker=style["marker"],
                    edgecolors="none",
                    linewidths=0.0,
                    s=95,
                    zorder=6,
                )
                ax.axhline(float(theta_true[param_idx]), color="0.35", linestyle="--", linewidth=1.2)
                ax.set_ylabel(rf"$\theta_{param_idx + 1}$")
                ax.grid(True, which="major", alpha=0.28, linewidth=0.8)

        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlim(left_limit, global_max_time * 1.03)

        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        return _save_figure(fig, output_path)


def make_lv_corruption_theta_vs_time_plot(path_map, output_path=None, se_scale=1.96, init_time_factor=0.85):
    line_styles = {"c15": ("--", "-"), "c35": (":", "-.")}

    with plt.rc_context(LOCAL_PLOT_RC):
        fig, axes = plt.subplots(2, 1, figsize=(6.3, 5.4), dpi=150, sharex=True)
        theta_true = None
        theta0_shared = None
        shared_init_time = None
        global_max_time = 0.0
        left_limit = np.inf

        for label, paths in path_map.items():
            sgd_time = []
            pgd_time = []
            sgd_theta = []
            pgd_theta = []
            sgd_theta_se = []
            pgd_theta_se = []
            theta0 = None

            for npz_path in paths:
                data = _load_npz_dict(npz_path)
                theta_true = np.asarray(data["theta_true"], dtype=float)
                theta0 = np.asarray(data["theta0"], dtype=float)
                theta0_shared = theta0
                sgd_time.append(float(np.asarray(data["sgd_elapsed_mean"], dtype=float)))
                pgd_time.append(float(np.asarray(data["pgd_elapsed_mean"], dtype=float)))
                sgd_theta.append(np.asarray(data["sgd_theta_mean"], dtype=float)[:2])
                pgd_theta.append(np.asarray(data["pgd_theta_mean"], dtype=float)[:2])
                sgd_theta_se.append(np.asarray(data["sgd_theta_se"], dtype=float)[:2])
                pgd_theta_se.append(np.asarray(data["pgd_theta_se"], dtype=float)[:2])

            sgd_time = np.asarray(sgd_time, dtype=float)
            pgd_time = np.asarray(pgd_time, dtype=float)
            sgd_theta = np.asarray(sgd_theta, dtype=float)
            pgd_theta = np.asarray(pgd_theta, dtype=float)
            sgd_theta_se = np.asarray(sgd_theta_se, dtype=float)
            pgd_theta_se = np.asarray(pgd_theta_se, dtype=float)
            if len(sgd_time) == 0 or len(pgd_time) == 0:
                continue

            global_max_time = max(global_max_time, float(np.max(sgd_time)), float(np.max(pgd_time)))
            sgd_order = np.argsort(sgd_time)
            pgd_order = np.argsort(pgd_time)
            init_time = init_time_factor * min(float(np.min(sgd_time)), float(np.min(pgd_time)))
            left_limit = min(left_limit, 0.8 * init_time)
            if shared_init_time is None:
                shared_init_time = init_time
            line_init_time = shared_init_time
            sgd_ls, pgd_ls = line_styles.get(label, ("--", "-"))

            for param_idx, ax in enumerate(axes):
                sgd_plot_time = np.concatenate(([line_init_time], sgd_time[sgd_order]))
                sgd_plot_theta = np.concatenate(([theta0[param_idx]], sgd_theta[sgd_order, param_idx]))
                pgd_plot_time = np.concatenate(([line_init_time], pgd_time[pgd_order]))
                pgd_plot_theta = np.concatenate(([theta0[param_idx]], pgd_theta[pgd_order, param_idx]))

                ax.plot(
                    sgd_plot_time,
                    sgd_plot_theta,
                    color=COLORS["gd"],
                    linestyle=sgd_ls,
                    linewidth=2.0,
                    marker=None,
                )
                ax.fill_between(
                    sgd_time[sgd_order],
                    sgd_theta[sgd_order, param_idx] - se_scale * sgd_theta_se[sgd_order, param_idx],
                    sgd_theta[sgd_order, param_idx] + se_scale * sgd_theta_se[sgd_order, param_idx],
                    color=COLORS["gd"],
                    alpha=0.10,
                )
                ax.plot(
                    pgd_plot_time,
                    pgd_plot_theta,
                    color=COLORS["pgd"],
                    linestyle=pgd_ls,
                    linewidth=2.0,
                    marker=None,
                )
                ax.fill_between(
                    pgd_time[pgd_order],
                    pgd_theta[pgd_order, param_idx] - se_scale * pgd_theta_se[pgd_order, param_idx],
                    pgd_theta[pgd_order, param_idx] + se_scale * pgd_theta_se[pgd_order, param_idx],
                    color=COLORS["pgd"],
                    alpha=0.10,
                )
                ax.axhline(float(theta_true[param_idx]), color="0.35", linestyle="--", linewidth=1.2)
                ax.set_ylabel(rf"$\theta_{param_idx + 1}$")
                ax.grid(True, which="major", alpha=0.28, linewidth=0.8)

        if theta0_shared is not None and shared_init_time is not None:
            for param_idx, ax in enumerate(axes):
                ax.scatter(
                    shared_init_time,
                    theta0_shared[param_idx],
                    color="#4D4D4D",
                    marker="o",
                    edgecolors="none",
                    linewidths=0.0,
                    s=95,
                    zorder=6,
                )

        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlim(left_limit, global_max_time * 1.03)

        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        return _save_figure(fig, output_path)


def make_lv_theta_error_heatmap(results_dir, output_path):
    results_dir = Path(results_dir)
    filename_pattern = re.compile(
        r"lotka_volterra_lengthscale_regularization_grid_pgd_ell_min_"
        r"(?P<ell>[^_]+)_lambda_(?P<lam>[^_]+)\.npz$"
    )

    error_by_cell = {}
    ell_values = set()
    lambda_values = set()

    for npz_path in sorted(results_dir.glob("lotka_volterra_lengthscale_regularization_grid_pgd_ell_min_*_lambda_*.npz")):
        match = filename_pattern.match(npz_path.name)
        if match is None:
            continue

        ell = _parse_heatmap_float(match.group("ell"))
        lam = _parse_heatmap_float(match.group("lam"))
        data = _load_npz_dict(npz_path)
        theta_true = np.asarray(data["theta_true"], dtype=float)
        theta_final = np.asarray(data["pgd_theta_mean"], dtype=float)
        theta_error = float(np.linalg.norm(theta_final - theta_true))

        error_by_cell[(lam, ell)] = theta_error
        ell_values.add(ell)
        lambda_values.add(lam)

    if not error_by_cell:
        raise ValueError(f"No LV heatmap files found in {results_dir}")

    sorted_ells = np.asarray(sorted(ell_values), dtype=float)
    sorted_lambdas = np.asarray(sorted(lambda_values), dtype=float)
    error_grid = np.full((len(sorted_lambdas), len(sorted_ells)), np.nan, dtype=float)

    for i, lam in enumerate(sorted_lambdas):
        for j, ell in enumerate(sorted_ells):
            if (float(lam), float(ell)) in error_by_cell:
                error_grid[i, j] = error_by_cell[(float(lam), float(ell))]

    return _draw_theta_error_heatmap(error_grid, sorted_ells, sorted_lambdas, output_path)


def make_lv_lengthscale_pair_heatmaps(results_dir, theta_output_path=None, mmd_output_path=None):
    results_dir = Path(results_dir)
    summary_path = (
        results_dir / "lotka_volterra_lengthscale_regularization_grid_pgd_ell_min_pgd_ell0_summary.npz"
    )
    if not summary_path.exists():
        raise ValueError(f"Missing LV pair-summary file: {summary_path}")

    summary = _load_npz_dict(summary_path)
    ell_inf_values = np.asarray(summary["lengthscale_values"], dtype=float)
    ell0_values = np.asarray(summary["secondary_lengthscale_values"], dtype=float)
    lambda_values = np.asarray(summary["lambda_scales"], dtype=float)
    output_paths = np.asarray(summary["output_paths"], dtype=object)
    eval_grids = np.asarray(summary["pgd_eval_mean_grid"], dtype=float)

    theta_error_grids = np.full_like(eval_grids, np.nan, dtype=float)
    for lambda_idx in range(output_paths.shape[0]):
        for ell0_idx in range(output_paths.shape[1]):
            for ell_idx in range(output_paths.shape[2]):
                npz_path = Path(str(output_paths[lambda_idx, ell0_idx, ell_idx]))
                data = _load_npz_dict(npz_path)
                theta_true = np.asarray(data["theta_true"], dtype=float)
                theta_final = np.asarray(data["pgd_theta_mean"], dtype=float)
                theta_error_grids[lambda_idx, ell0_idx, ell_idx] = float(
                    np.linalg.norm(theta_final - theta_true)
                )

    theta_fig = _draw_lengthscale_pair_heatmaps(
        grids=[theta_error_grids[idx] for idx in range(len(lambda_values))],
        x_values=ell0_values,
        y_values=ell_inf_values,
        lambda_values=lambda_values,
        colorbar_label=r"$\|\bar{\theta}_{final} - \theta_{true}\|_2$",
        output_path=theta_output_path,
    )
    mmd_fig = _draw_lengthscale_pair_heatmaps(
        grids=[eval_grids[idx] for idx in range(len(lambda_values))],
        x_values=ell0_values,
        y_values=ell_inf_values,
        lambda_values=lambda_values,
        colorbar_label=r"Final MMD",
        output_path=mmd_output_path,
    )
    return theta_fig, mmd_fig


def make_lv_step_size_boxplot(results_dir, output_path=None, metric="pgd_eval_losses"):
    results_dir = Path(results_dir)
    filename_pattern = re.compile(
        r"lotka_volterra_step_size_ablation_pgd_gamma_sweep_(?P<gamma>[^_]+)\.npz$"
    )

    gamma_values = []
    box_values = []

    for npz_path in sorted(results_dir.glob("lotka_volterra_step_size_ablation_pgd_gamma_sweep_*.npz")):
        if npz_path.name.endswith("_summary.npz"):
            continue
        match = filename_pattern.match(npz_path.name)
        if match is None:
            continue

        gamma = _parse_heatmap_float(match.group("gamma"))
        data = _load_npz_dict(npz_path)

        if metric == "pgd_eval_losses":
            values = np.asarray(data["pgd_eval_losses"], dtype=float)
            ylabel = "Final MMD"
        elif metric == "pgd_theta_error":
            theta_true = np.asarray(data["theta_true"], dtype=float)
            theta_finals = np.asarray(data["pgd_theta_finals"], dtype=float)
            values = np.linalg.norm(theta_finals - theta_true[None, :], axis=1)
            ylabel = r"$\|\theta_{final} - \theta_{true}\|_2$"
        else:
            raise ValueError("metric must be 'pgd_eval_losses' or 'pgd_theta_error'")

        gamma_values.append(gamma)
        box_values.append(values)

    if not box_values:
        raise ValueError(f"No LV step-size ablation files found in {results_dir}")

    order = np.argsort(np.asarray(gamma_values, dtype=float))
    gamma_values = [gamma_values[idx] for idx in order]
    box_values = [box_values[idx] for idx in order]

    with plt.rc_context(LOCAL_PLOT_RC):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=SUMMARY_DPI)
        bp = ax.boxplot(box_values, patch_artist=True, widths=0.6)

        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS["pgd"])
            patch.set_alpha(0.75)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(2.0)

        ax.set_xticks(np.arange(1, len(gamma_values) + 1))
        ax.set_xticklabels([f"{gamma:g}" for gamma in gamma_values])
        ax.set_xlabel("Step size")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return _save_figure(fig, output_path, bbox_inches="tight")


def make_mmd_flow_figure(
    comparison_npz,
    mmd_vs_n_npz_paths,
    mmd_vs_n_ns,
    mmd_vs_iteration_npz,
    lhs_rhs_npz,
    output_path=None,
    se_scale=1.96,
):
    _set_plot_style("mmd_flow")
    return make_four_panel_figure(
        comparison_npz=comparison_npz,
        mmd_vs_n_npz_paths=mmd_vs_n_npz_paths,
        mmd_vs_n_ns=mmd_vs_n_ns,
        mmd_vs_iteration_npz=mmd_vs_iteration_npz,
        lhs_rhs_npz=lhs_rhs_npz,
        output_path=output_path,
        se_scale=se_scale,
    )


def _theta_error_series_gk(npz_path):
    data = _load_npz_dict(npz_path)
    theta_true = np.asarray(data["theta_true"], dtype=float)
    theta_finals = np.asarray(data["adaptive_thetas"], dtype=float)
    return np.linalg.norm(theta_finals - theta_true[None, :], axis=1)


def _theta_error_series_lv(npz_path):
    data = _load_npz_dict(npz_path)
    theta_true = np.asarray(data["theta_true"], dtype=float)
    theta_finals = np.asarray(data["pgd_theta_finals"], dtype=float)
    return np.linalg.norm(theta_finals - theta_true[None, :], axis=1)


def _collect_gk_series_from_pattern(results_dir, pattern):
    collected = []
    for npz_path in sorted(Path(results_dir).rglob("g_n_k_fixed*.npz")):
        match = pattern.search(npz_path.as_posix())
        if match is None:
            continue
        value = float(match.group(1).replace("p", ".").replace("m", "-"))
        collected.append((value, _theta_error_series_gk(npz_path)))
    collected.sort(key=lambda item: item[0])
    return collected


def _collect_lv_series_from_filename_glob(results_dir, glob_pattern, filename_pattern):
    collected = []
    for npz_path in sorted(Path(results_dir).glob(glob_pattern)):
        if npz_path.name.endswith("_summary.npz"):
            continue
        match = filename_pattern.match(npz_path.name)
        if match is None:
            continue
        value = float(match.group(1).replace("p", ".").replace("m", "-"))
        collected.append((value, _theta_error_series_lv(npz_path)))
    collected.sort(key=lambda item: item[0])
    return collected


def _collect_gk_mn_theta_errors(results_dir):
    pattern = re.compile(r"g_and_k_observation_model_grid_m_(?P<m>[^_]+)_n_(?P<n>[^/]+)/")
    by_mn = {}
    m_values = set()
    n_values = set()
    for npz_path in sorted(Path(results_dir).rglob("g_n_k_fixed*.npz")):
        match = pattern.search(npz_path.as_posix())
        if match is None:
            continue
        m = float(match.group("m").replace("p", ".").replace("m", "-"))
        n = float(match.group("n").replace("p", ".").replace("m", "-"))
        by_mn[(m, n)] = _theta_error_series_gk(npz_path)
        m_values.add(m)
        n_values.add(n)
    return by_mn, sorted(m_values), sorted(n_values)


def _collect_lv_mn_theta_errors(results_dir):
    pattern = re.compile(r"lotka_volterra_observation_model_grid_m_(?P<m>[^_]+)_n_(?P<n>[^_]+)\.npz$")
    by_mn = {}
    m_values = set()
    n_values = set()
    for npz_path in sorted(Path(results_dir).glob("lotka_volterra_observation_model_grid_m_*_n_*.npz")):
        match = pattern.match(npz_path.name)
        if match is None:
            continue
        m = float(match.group("m").replace("p", ".").replace("m", "-"))
        n = float(match.group("n").replace("p", ".").replace("m", "-"))
        by_mn[(m, n)] = _theta_error_series_lv(npz_path)
        m_values.add(m)
        n_values.add(n)
    return by_mn, sorted(m_values), sorted(n_values)


def _draw_gk_step_boxplot(ax, results_dir):
    pattern = re.compile(r"g_and_k_step_size_ablation_gamma_pgd0_sweep_([^/]+)/")
    collected = {}
    summary_path = Path(results_dir) / "g_and_k_step_size_ablation_gamma_pgd0_sweep_summary.npz"
    if summary_path.exists():
        summary = _load_npz_dict(summary_path)
        if "output_paths" in summary and "sweep_values" in summary:
            for gamma, npz_path in zip(summary["sweep_values"], summary["output_paths"]):
                npz_path = Path(npz_path)
                if npz_path.exists():
                    collected[float(gamma)] = _theta_error_series_gk(npz_path)
    for gamma, values in _collect_gk_series_from_pattern(results_dir, pattern):
        collected[gamma] = values
    ordered = sorted(collected.items(), key=lambda item: item[0])
    _make_single_color_boxplot(
        ax,
        [values for _, values in ordered],
        [f"{gamma:g}" for gamma, _ in ordered],
        COLORS["pgd"],
        "step size," r" $\gamma$",
        r"$\|\theta_{\mathrm{final}}-\theta_{\mathrm{true}}\|_2$",
        logy=True,
    )


def _draw_gk_decay_boxplot(ax, results_dir):
    pattern = re.compile(r"g_and_k_decay_ablation_decay_sweep_([^/]+)/")
    collected = _collect_gk_series_from_pattern(results_dir, pattern)
    _make_single_color_boxplot(
        ax,
        [values for _, values in collected],
        [f"{value:g}" for value, _ in collected],
        COLORS["pgd"],
        "lengthscale decay rate",
        "",
        logy=True,
    )


def _draw_gk_mn_boxplot(ax, results_dir):
    by_mn, m_values, n_values = _collect_gk_mn_theta_errors(results_dir)
    grouped_by_n = [np.concatenate([by_mn[(m, n)] for m in m_values]) for n in n_values]
    _make_single_color_boxplot(
        ax,
        grouped_by_n,
        [f"{n:g}" for n in n_values],
        "#4c78a8",
        r"$n$",
        "",
        logy=True,
    )


def _draw_gk_ridge_boxplot(ax, results_dir):
    pattern = re.compile(r"g_and_k_regularization_ablation_lambda_scale_sweep_([^/]+)/")
    collected = {}
    summary_path = Path(results_dir) / "g_and_k_regularization_ablation_lambda_scale_sweep_summary.npz"
    if summary_path.exists():
        summary = _load_npz_dict(summary_path)
        if "output_paths" in summary and "sweep_values" in summary:
            for lam, npz_path in zip(summary["sweep_values"], summary["output_paths"]):
                npz_path = Path(npz_path)
                if npz_path.exists():
                    collected[float(lam)] = _theta_error_series_gk(npz_path)
    for npz_path in sorted(Path(results_dir).rglob("g_n_k_fixed*.npz")):
        match = pattern.search(npz_path.as_posix())
        if match is None:
            continue
        lam = float(match.group(1).replace("p", ".").replace("m", "-"))
        collected[lam] = _theta_error_series_gk(npz_path)
    ordered = sorted(collected.items(), key=lambda item: item[0])
    if not ordered:
        raise FileNotFoundError(f"No G-and-K ridge ablation .npz files found in {results_dir}")
    _make_single_color_boxplot(
        ax,
        [values for _, values in ordered],
        [f"{lam:g}" for lam, _ in ordered],
        COLORS["pgd"],
        r"ridge constant, $\lambda$",
        "",
        logy=True,
    )


def make_gk_ablation_summary(gk_step_dir, gk_decay_dir, gk_mn_dir, gk_ridge_dir, output_path=None, figsize=(24, 5), dpi=150):
    _set_plot_style("gnk_ablation")
    with plt.rc_context(LOCAL_PLOT_RC):
        fig, axes = plt.subplots(1, 4, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.18]})
        _draw_gk_step_boxplot(axes[0], gk_step_dir)
        _draw_gk_decay_boxplot(axes[1], gk_decay_dir)
        _draw_gk_mn_boxplot(axes[2], gk_mn_dir)
        _draw_gk_ridge_boxplot(axes[3], gk_ridge_dir)
        for ax in axes[1:]:
            ax.set_ylabel("")
        fig.tight_layout()
        return _save_figure(fig, output_path, bbox_inches="tight")


def _draw_lv_step_boxplot(ax, results_dir):
    collected = _collect_lv_series_from_filename_glob(
        results_dir,
        "lotka_volterra_step_size_ablation_pgd_gamma_sweep_*.npz",
        re.compile(r"lotka_volterra_step_size_ablation_pgd_gamma_sweep_([^_]+)\.npz$"),
    )
    _make_single_color_boxplot(
        ax,
        [values for _, values in collected],
        [f"{value:g}" for value, _ in collected],
        COLORS["pgd"],
        "step size," r" $\gamma$",
        r"$\|\theta_{\mathrm{final}}-\theta_{\mathrm{true}}\|_2$",
        logy=True,
    )


def _draw_lv_decay_boxplot(ax, results_dir):
    collected = _collect_lv_series_from_filename_glob(
        results_dir,
        "lotka_volterra_decay_ablation_pgd_decay_sweep_*.npz",
        re.compile(r"lotka_volterra_decay_ablation_pgd_decay_sweep_([^_]+)\.npz$"),
    )
    _make_single_color_boxplot(
        ax,
        [values for _, values in collected],
        [f"{value:g}" for value, _ in collected],
        COLORS["pgd"],
        "lengthscale decay rate",
        "",
        logy=True,
    )


def _draw_lv_mn_boxplot(ax, results_dir):
    by_mn, m_values, n_values = _collect_lv_mn_theta_errors(results_dir)
    grouped_by_n = [np.concatenate([by_mn[(m, n)] for m in m_values]) for n in n_values]
    _make_single_color_boxplot(
        ax,
        grouped_by_n,
        [f"{n:g}" for n in n_values],
        "#4c78a8",
        r"$n$",
        "",
        logy=True,
    )


def _draw_lv_ridge_boxplot(ax, results_dir):
    collected = {}
    summary_path = Path(results_dir) / "lotka_volterra_regularization_ablation_pgd_lambda_scale_sweep_summary.npz"
    if summary_path.exists():
        summary = _load_npz_dict(summary_path)
        if "output_paths" in summary and "sweep_values" in summary:
            for lam, npz_path in zip(summary["sweep_values"], summary["output_paths"]):
                npz_path = Path(npz_path)
                if npz_path.exists():
                    collected[float(lam)] = _theta_error_series_lv(npz_path)
    for lam, values in _collect_lv_series_from_filename_glob(
        results_dir,
        "lotka_volterra_regularization_ablation_pgd_lambda_scale_sweep_*.npz",
        re.compile(r"lotka_volterra_regularization_ablation_pgd_lambda_scale_sweep_([^_]+)\.npz$"),
    ):
        collected[lam] = values
    ordered = sorted(collected.items(), key=lambda item: item[0])
    if not ordered:
        raise FileNotFoundError(f"No LV ridge ablation .npz files found in {results_dir}")
    _make_single_color_boxplot(
        ax,
        [values for _, values in ordered],
        [f"{lam:g}" for lam, _ in ordered],
        COLORS["pgd"],
        r"ridge constant, $\lambda$",
        "",
        logy=True,
    )


def make_lv_ablation_summary(lv_step_dir, lv_decay_dir, lv_mn_dir, lv_ridge_dir, output_path=None, figsize=(24, 5), dpi=150):
    _set_plot_style("lv_ablation")
    with plt.rc_context(LOCAL_PLOT_RC):
        fig, axes = plt.subplots(1, 4, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.18]})
        _draw_lv_step_boxplot(axes[0], lv_step_dir)
        _draw_lv_decay_boxplot(axes[1], lv_decay_dir)
        _draw_lv_mn_boxplot(axes[2], lv_mn_dir)
        _draw_lv_ridge_boxplot(axes[3], lv_ridge_dir)
        for ax in axes[1:]:
            ax.set_ylabel("")
        fig.tight_layout()
        return _save_figure(fig, output_path, bbox_inches="tight")


def _theta_error_histories(data, histories_key, theta_true_key="theta_true"):
    theta_true = np.asarray(data[theta_true_key], dtype=float)
    histories = np.asarray(data[histories_key], dtype=float)
    return np.linalg.norm(histories - theta_true[None, None, :], axis=-1)


def _draw_gk_cross_method(ax, npz_path):
    data = _load_npz_dict(npz_path)
    baseline_err = _theta_error_histories(data, "baseline_theta_histories")
    adaptive_sgd_err = _theta_error_histories(data, "adaptive_sgd_theta_histories")
    pgd_err = _theta_error_histories(data, "adaptive_theta_histories")
    fixed_pgd_err = _theta_error_histories(data, "fixed_pgd_theta_histories")
    baseline_steps = np.asarray(data["baseline_history_steps"], dtype=float) + 1.0
    adaptive_sgd_steps = np.asarray(data["adaptive_sgd_history_steps"], dtype=float) + 1.0
    pgd_steps = np.asarray(data["adaptive_history_steps"], dtype=float) + 1.0
    fixed_pgd_steps = np.asarray(data["fixed_pgd_history_steps"], dtype=float) + 1.0
    ax.plot(baseline_steps, np.mean(baseline_err, axis=0), color=METHOD_COLORS["sgd"], label="GD (fixed $\ell$)")
    ax.plot(adaptive_sgd_steps, np.mean(adaptive_sgd_err, axis=0), color=METHOD_COLORS["adaptive_sgd"], label="GD (adaptive $\ell$)")
    ax.plot(pgd_steps, np.mean(pgd_err, axis=0), color=METHOD_COLORS["pgd"], label="PGD (adaptive $\ell$)")
    ax.plot(fixed_pgd_steps, np.mean(fixed_pgd_err, axis=0), color=METHOD_COLORS["fixed_pgd"], label="PGD (fixed $\ell$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|\theta_{\mathrm{final}}-\theta_{\mathrm{true}}\|_2$")
    ax.set_title("G-and-K distribution", pad=TITLE_PAD)
    ax.legend(loc="best", frameon=True)


def _draw_lv_cross_method(ax, npz_path, max_iteration=11000):
    data = _load_npz_dict(npz_path)
    sgd_err = _theta_error_histories(data, "sgd_theta_histories")
    adaptive_sgd_err = _theta_error_histories(data, "adaptive_sgd_theta_histories")
    pgd_err = _theta_error_histories(data, "pgd_theta_histories")
    fixed_pgd_err = _theta_error_histories(data, "fixed_pgd_theta_histories")
    sgd_steps = np.asarray(data["sgd_history_steps"], dtype=float) + 1.0
    adaptive_sgd_steps = np.asarray(data["adaptive_sgd_history_steps"], dtype=float) + 1.0
    pgd_steps = np.asarray(data["pgd_history_steps"], dtype=float) + 1.0
    fixed_pgd_steps = np.asarray(data["fixed_pgd_history_steps"], dtype=float) + 1.0

    def truncate(steps, errors):
        mask = steps <= float(max_iteration)
        return steps[mask], errors[:, mask]

    sgd_steps, sgd_err = truncate(sgd_steps, sgd_err)
    adaptive_sgd_steps, adaptive_sgd_err = truncate(adaptive_sgd_steps, adaptive_sgd_err)
    pgd_steps, pgd_err = truncate(pgd_steps, pgd_err)
    fixed_pgd_steps, fixed_pgd_err = truncate(fixed_pgd_steps, fixed_pgd_err)
    ax.plot(sgd_steps, np.mean(sgd_err, axis=0), color=METHOD_COLORS["sgd"], label="GD (fixed $\ell$)")
    ax.plot(adaptive_sgd_steps, np.mean(adaptive_sgd_err, axis=0), color=METHOD_COLORS["adaptive_sgd"], label="GD (adaptive $\ell$)")
    ax.plot(pgd_steps, np.mean(pgd_err, axis=0), color=METHOD_COLORS["pgd"], label="PGD (adaptive $\ell$)")
    ax.plot(fixed_pgd_steps, np.mean(fixed_pgd_err, axis=0), color=METHOD_COLORS["fixed_pgd"], label="PGD (fixed $\ell$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("")
    ax.set_title("Lotka--Volterra", pad=TITLE_PAD)
    ax.legend(loc="best", frameon=True)


def make_cross_method_summary(gk_compare_npz, lv_compare_npz, output_path=None, figsize=(16, 5), dpi=150, max_iteration=11000):
    _set_plot_style("cross_method")
    with plt.rc_context(LOCAL_PLOT_RC):
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        _draw_gk_cross_method(axes[0], gk_compare_npz)
        _draw_lv_cross_method(axes[1], lv_compare_npz, max_iteration=max_iteration)
        fig.tight_layout()
        return _save_figure(fig, output_path, bbox_inches="tight")


def collect_gnk_time_series(paths):
    method_specs = [
        ("baseline", "GD", COLORS["gd"]),
        ("natural", "PGD (Briol et al.)", COLORS["natural"]),
        ("adaptive", "PGD (ours)", COLORS["pgd"]),
    ]
    series = []
    for prefix, label, color in method_specs:
        times = []
        mmds = []
        for npz_path in paths:
            data = _load_npz_dict(npz_path)
            times.append(float(np.asarray(data[f"{prefix}_elapsed_mean"], dtype=float)))
            mmds.append(np.sqrt(float(np.asarray(data[f"{prefix}_eval_mean"], dtype=float))))
        times = np.asarray(times, dtype=float)
        mmds = np.asarray(mmds, dtype=float)
        order = np.argsort(times)
        series.append({"label": label, "color": color, "time": times[order], "mmd": mmds[order]})
    return series


def _collect_lv_method_from_checkpoints(data, prefix, se_scale):
    required = [
        f"{prefix}_checkpoint_iterations",
        f"{prefix}_checkpoint_elapsed_mean",
        f"{prefix}_checkpoint_theta_mean",
    ]
    if any(key not in data for key in required):
        return None
    time = np.asarray(data[f"{prefix}_checkpoint_elapsed_mean"], dtype=float)
    theta = np.asarray(data[f"{prefix}_checkpoint_theta_mean"], dtype=float)[..., :2]
    theta_std_key = f"{prefix}_checkpoint_theta_std"
    if theta_std_key in data:
        theta_se = se_scale * np.asarray(data[theta_std_key], dtype=float)[..., :2]
    else:
        theta_se = np.zeros_like(theta)
    iterations = np.asarray(data[f"{prefix}_checkpoint_iterations"], dtype=np.int32)
    order = np.argsort(time)
    return {
        "iterations": iterations[order],
        "time": time[order],
        "theta": theta[order],
        "theta_se": theta_se[order],
    }


def collect_lv_single_time_record(label, npz_path, se_scale, init_time_factor, style=None, linestyles=None):
    data = _load_npz_dict(npz_path)
    theta_true = np.asarray(data["theta_true"], dtype=float)
    theta0 = np.asarray(data["theta0"], dtype=float)
    sgd = _collect_lv_method_from_checkpoints(data, "sgd", se_scale)
    natural = _collect_lv_method_from_checkpoints(data, "natural", se_scale)
    pgd = _collect_lv_method_from_checkpoints(data, "pgd", se_scale)
    if sgd is None or pgd is None:
        raise KeyError(f"{npz_path} is missing LV checkpoint data for SGD and/or PGD.")
    time_arrays = [sgd["time"], pgd["time"]]
    if natural is not None:
        time_arrays.append(natural["time"])
    min_time = min(float(np.min(arr)) for arr in time_arrays if arr.size)
    max_time = max(float(np.max(arr)) for arr in time_arrays if arr.size)
    return {
        "label": label,
        "style": style or {},
        "linestyles": linestyles or {"sgd": "-", "natural": "-", "pgd": "-"},
        "has_natural": natural is not None,
        "theta_true": theta_true,
        "theta0": theta0,
        "init_time": init_time_factor * min_time,
        "left_limit": 0.8 * init_time_factor * min_time,
        "max_time": max_time,
        "sgd_time": sgd["time"],
        "pgd_time": pgd["time"],
        "sgd_theta": sgd["theta"],
        "pgd_theta": pgd["theta"],
        "sgd_theta_se": sgd["theta_se"],
        "pgd_theta_se": pgd["theta_se"],
        "natural_time": None if natural is None else natural["time"],
        "natural_theta": None if natural is None else natural["theta"],
        "natural_theta_se": None if natural is None else natural["theta_se"],
    }


def collect_lv_time_series(path_map, se_scale=1.96, init_time_factor=0.85):
    init_styles = {
        "50, 60": {"marker": "o", "init_color": "#7B3294"},
        "90, 90": {"marker": "^", "init_color": "#4D4D4D"},
    }
    records = []
    global_max_time = 0.0
    left_limit = np.inf
    theta_true = None
    for init_label, npz_path in path_map.items():
        record = collect_lv_single_time_record(
            label=init_label,
            npz_path=npz_path,
            se_scale=se_scale,
            init_time_factor=init_time_factor,
            style=init_styles.get(init_label, {"marker": "o", "init_color": "#4D4D4D"}),
        )
        records.append(record)
        theta_true = record["theta_true"]
        global_max_time = max(global_max_time, record["max_time"])
        left_limit = min(left_limit, record["left_limit"])
    return {"records": records, "theta_true": theta_true, "left_limit": left_limit, "global_max_time": global_max_time}


def collect_lv_corruption_time_series(path_map, se_scale=1.96, init_time_factor=0.85):
    corruption_styles = {
        "15%": {"sgd": CORRUPTION_LINESTYLES["15"], "natural": CORRUPTION_LINESTYLES["15"], "pgd": CORRUPTION_LINESTYLES["15"]},
        "35%": {"sgd": CORRUPTION_LINESTYLES["35"], "natural": CORRUPTION_LINESTYLES["35"], "pgd": CORRUPTION_LINESTYLES["35"]},
    }
    records = []
    global_max_time = 0.0
    left_limit = np.inf
    theta_true = None
    for label, npz_path in path_map.items():
        record = collect_lv_single_time_record(
            label=label,
            npz_path=npz_path,
            se_scale=se_scale,
            init_time_factor=init_time_factor,
            style={"marker": "o", "init_color": "#4D4D4D"},
            linestyles=corruption_styles.get(label, {"sgd": "-", "natural": "-", "pgd": "-"}),
        )
        records.append(record)
        theta_true = record["theta_true"]
        global_max_time = max(global_max_time, record["max_time"])
        left_limit = min(left_limit, record["left_limit"])
    return {"records": records, "theta_true": theta_true, "left_limit": left_limit, "global_max_time": global_max_time}


def draw_lv_time_axes(lv_axes, lv_series, title, show_natural=True):
    for record in lv_series["records"]:
        for param_idx, ax in enumerate(lv_axes):
            sgd_time = np.concatenate(([record["init_time"]], record["sgd_time"]))
            sgd_theta = np.concatenate(([record["theta0"][param_idx]], record["sgd_theta"][:, param_idx]))
            pgd_time = np.concatenate(([record["init_time"]], record["pgd_time"]))
            pgd_theta = np.concatenate(([record["theta0"][param_idx]], record["pgd_theta"][:, param_idx]))
            ax.plot(sgd_time, sgd_theta, color=COLORS["gd"], linestyle=record["linestyles"]["sgd"], linewidth=2.0)
            ax.fill_between(
                record["sgd_time"],
                record["sgd_theta"][:, param_idx] - record["sgd_theta_se"][:, param_idx],
                record["sgd_theta"][:, param_idx] + record["sgd_theta_se"][:, param_idx],
                color=COLORS["gd"], alpha=0.10, linewidth=0,
            )
            if show_natural and record.get("has_natural", False):
                natural_time = np.concatenate(([record["init_time"]], record["natural_time"]))
                natural_theta = np.concatenate(([record["theta0"][param_idx]], record["natural_theta"][:, param_idx]))
                ax.plot(natural_time, natural_theta, color=COLORS["natural"], linestyle=record["linestyles"].get("natural", "-"), linewidth=2.0)
                ax.fill_between(
                    record["natural_time"],
                    record["natural_theta"][:, param_idx] - record["natural_theta_se"][:, param_idx],
                    record["natural_theta"][:, param_idx] + record["natural_theta_se"][:, param_idx],
                    color=COLORS["natural"], alpha=0.10, linewidth=0,
                )
            ax.plot(pgd_time, pgd_theta, color=COLORS["pgd"], linestyle=record["linestyles"]["pgd"], linewidth=2.0)
            ax.fill_between(
                record["pgd_time"],
                record["pgd_theta"][:, param_idx] - record["pgd_theta_se"][:, param_idx],
                record["pgd_theta"][:, param_idx] + record["pgd_theta_se"][:, param_idx],
                color=COLORS["pgd"], alpha=0.10, linewidth=0,
            )
            ax.axhline(float(lv_series["theta_true"][param_idx]), color="0.35", linestyle="--", linewidth=1.2)
            ax.set_ylabel(rf"$\theta_{param_idx + 1}$")
            ax.grid(True, which="major", alpha=0.24, linewidth=0.8)
    for ax in lv_axes:
        ax.set_xscale("log")
        ax.set_xlim(lv_series["left_limit"], lv_series["global_max_time"] * 1.03)
    lv_axes[0].set_title(title, fontsize=21, pad=10)
    lv_axes[0].tick_params(labelbottom=False)
    lv_axes[-1].set_xlabel("Time (s)")


def make_combined_time_summary(gnk_series, lv_series, lv_corruption_series, output_path=None):
    _set_plot_style("combined_time")
    with plt.rc_context(PLOT_RC):
        fig = plt.figure(figsize=(22.4, 5), dpi=150)
        grid = fig.add_gridspec(2, 3, width_ratios=[1.08, 1.0, 1.0], wspace=0.34, hspace=0.18)
        ax_gnk = fig.add_subplot(grid[:, 0])
        lv_axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])]
        lv_corruption_axes = [fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 2])]
        for series in gnk_series:
            ax_gnk.plot(series["time"], series["mmd"], color=series["color"], marker="o", linewidth=2.8, markersize=6.5, label=series["label"])
        ax_gnk.set_xscale("log")
        ax_gnk.set_yscale("log")
        ax_gnk.set_xlabel("Time (s)")
        ax_gnk.set_ylabel(r"$\mathrm{MMD}_{\ell_\infty}(\mathbb{P}_\theta, \mathbb{Q})$")
        ax_gnk.set_title("G-and-K", fontsize=21, pad=10)
        ax_gnk.grid(True, which="major", alpha=0.22, linewidth=0.8)
        draw_lv_time_axes(lv_axes, lv_series, "Lotka-Volterra")
        draw_lv_time_axes(lv_corruption_axes, lv_corruption_series, "Lotka-Volterra (corruption)", show_natural=True)
        for ax in lv_corruption_axes:
            upper = ax.get_ylim()[1]
            ax.set_ylim(bottom=50.0, top=upper)
        top_handles = [
            Line2D([0], [0], color=COLORS["gd"], lw=2.8, marker="o", label="GD"),
            Line2D([0], [0], color=COLORS["natural"], lw=2.8, marker="o", label="PGD (Briol et al.)"),
            Line2D([0], [0], color=COLORS["pgd"], lw=2.8, marker="o", label="PGD (ours)"),
            Line2D([0], [0], color="black", lw=2.2, linestyle=CORRUPTION_LINESTYLES["0"], label="0\\%"),
            Line2D([0], [0], color="black", lw=2.2, linestyle=CORRUPTION_LINESTYLES["15"], label="15\\%"),
            Line2D([0], [0], color="black", lw=2.2, linestyle=CORRUPTION_LINESTYLES["35"], label="35\\%"),
        ]
        fig.legend(top_handles, [handle.get_label() for handle in top_handles], loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=6, frameon=False, fontsize=24, columnspacing=1.1, handletextpad=0.6)
        return _save_figure(fig, output_path, bbox_inches="tight")


def _run_if_inputs_exist(label, input_paths, output_path, plotter):
    missing_inputs = [Path(path) for path in input_paths if not Path(path).exists()]
    if missing_inputs:
        print(
            f"Skipping {label} because {len(missing_inputs)} input file(s) are missing: "
            + ", ".join(str(path) for path in missing_inputs)
        )
        return None

    fig = plotter()
    print(f"Saved {label} to {output_path}")
    plt.close(fig)
    return fig


def run_mmd_flow(root, figures_dir):
    _set_plot_style("mmd_flow")
    nonparametric_dir = root / "results" / "nonparametric"
    mmd_comparison_npz = nonparametric_dir / "results_n100f.npz"
    mmd_vs_n_npz_paths = [
        nonparametric_dir / "results_n10f.npz",
        nonparametric_dir / "results_n30f.npz",
        nonparametric_dir / "results_n100f.npz",
        nonparametric_dir / "results_n300f.npz",
    ]
    mmd_iteration_npz = nonparametric_dir / "results_n100f.npz"
    mmd_lhs_rhs_npz = nonparametric_dir / "results_n100f.npz"
    mmd_output = figures_dir / "mmd_flow.pdf"
    _run_if_inputs_exist(
        "MMD flow figure",
        [mmd_comparison_npz, *mmd_vs_n_npz_paths, mmd_iteration_npz, mmd_lhs_rhs_npz],
        mmd_output,
        lambda: make_four_panel_figure(
            comparison_npz=mmd_comparison_npz,
            mmd_vs_n_npz_paths=mmd_vs_n_npz_paths,
            mmd_vs_n_ns=[10, 30, 100, 300],
            mmd_vs_iteration_npz=mmd_iteration_npz,
            lhs_rhs_npz=mmd_lhs_rhs_npz,
            output_path=mmd_output,
            se_scale=1.96,
        ),
    )


def run_gnk(root, figures_dir):
    _set_plot_style("gnk")
    gnk_dir = root / "results" / "gnk"
    gk_mmd_npz = gnk_dir / "g_n_k_theta0_3p500_2p000_0p600_m0p800.npz"
    gk_trajectory_npz_paths = [
        gnk_dir / "g_n_k_theta0_3p500_2p000_0p600_m0p800.npz",
        gnk_dir / "g_n_k_theta0_2p000_2p000_1p300_m0p600.npz",
    ]
    gk_lhs_rhs_npz = gnk_dir / "g_n_k_theta0_3p500_2p000_0p600_m0p800.npz"
    gk_summary_output = figures_dir / "gk_summary.pdf"
    _run_if_inputs_exist(
        "G-and-K summary figure",
        [gk_mmd_npz, *gk_trajectory_npz_paths, gk_lhs_rhs_npz],
        gk_summary_output,
        lambda: make_gk_summary_figure(
            mmd_npz_path=gk_mmd_npz,
            trajectory_npz_paths=gk_trajectory_npz_paths,
            lhs_rhs_npz_path=gk_lhs_rhs_npz,
            output_path=gk_summary_output,
            show_method_legend=False,
            show_lhs_rhs_legend=True,
        ),
    )


def run_lv(root, figures_dir):
    _set_plot_style("lv")
    lv_dir = root / "results" / "lv"
    lv_clean_npz_paths = [
        lv_dir / "lotka_volterra_results_50_60.npz",
        lv_dir / "lotka_volterra_results_90_90.npz",
    ]
    lv_corruption_npz_paths = [
        lv_dir / "lotka_volterra_results_60_60_c15.npz",
        lv_dir / "lotka_volterra_results_60_60_c35.npz",
    ]
    lv_summary_output = figures_dir / "lv_summary.pdf"
    _run_if_inputs_exist(
        "LV summary figure",
        [*lv_clean_npz_paths, *lv_corruption_npz_paths],
        lv_summary_output,
        lambda: make_lv_five_panel_summary(
            clean_npz_paths=lv_clean_npz_paths,
            corruption_npz_paths=lv_corruption_npz_paths,
            output_path=lv_summary_output,
            figsize=SUMMARY_FIGSIZE,
            dpi=DPI,
            se_scale=1.96,
        ),
    )

def run_gnk_ablations(root, figures_dir):
    _set_plot_style("gnk_ablation")
    ablations_dir = root / "ablations"
    gk_step_dir = ablations_dir / "gk_gamma"
    gk_decay_dir = ablations_dir / "gk_decay"
    gk_mn_dir = ablations_dir / "gk_mn_grid"
    gk_ridge_dir = ablations_dir / "gk_ridge"
    gk_ablation_output = figures_dir / "gk_ablation_summary.pdf"
    input_paths = [
        *list(gk_step_dir.rglob("g_n_k_fixed*.npz")),
        *list(gk_decay_dir.rglob("g_n_k_fixed*.npz")),
        *list(gk_mn_dir.rglob("g_n_k_fixed*.npz")),
        *list(gk_ridge_dir.rglob("g_n_k_fixed*.npz")),
    ]
    _run_if_inputs_exist(
        "G-and-K ablation summary",
        input_paths,
        gk_ablation_output,
        lambda: make_gk_ablation_summary(
            gk_step_dir=gk_step_dir,
            gk_decay_dir=gk_decay_dir,
            gk_mn_dir=gk_mn_dir,
            gk_ridge_dir=gk_ridge_dir,
            output_path=gk_ablation_output,
            figsize=(24, 5),
            dpi=SUMMARY_DPI,
        ),
    )


def run_lv_ablations(root, figures_dir):
    _set_plot_style("lv_ablation")
    ablations_dir = root / "ablations"
    lv_step_dir = ablations_dir / "lv_gamma"
    lv_decay_dir = ablations_dir / "lv_decay"
    lv_mn_dir = ablations_dir / "lv_mn_grid"
    lv_ridge_dir = ablations_dir / "lv_ridge"
    lv_ablation_output = figures_dir / "lv_ablation_summary.pdf"
    input_paths = [
        *list(lv_step_dir.glob("lotka_volterra_step_size_ablation_pgd_gamma_sweep_*.npz")),
        *list(lv_decay_dir.glob("lotka_volterra_decay_ablation_pgd_decay_sweep_*.npz")),
        *list(lv_mn_dir.glob("lotka_volterra_observation_model_grid_m_*_n_*.npz")),
        *list(lv_ridge_dir.glob("lotka_volterra_regularization_ablation_pgd_lambda_scale_sweep_*.npz")),
    ]
    _run_if_inputs_exist(
        "LV ablation summary",
        input_paths,
        lv_ablation_output,
        lambda: make_lv_ablation_summary(
            lv_step_dir=lv_step_dir,
            lv_decay_dir=lv_decay_dir,
            lv_mn_dir=lv_mn_dir,
            lv_ridge_dir=lv_ridge_dir,
            output_path=lv_ablation_output,
            figsize=(24, 5),
            dpi=SUMMARY_DPI,
        ),
    )


def run_cross_method(root, figures_dir):
    _set_plot_style("cross_method")
    gk_compare_npz = root / "cross_method" / "gnk" / "g_n_k_fixed600_theta0_2p000_2p000_1p500_m0p300.npz"
    lv_compare_npz = root / "cross_method" / "lv" / "lotka_volterra_results_pgd_vs_sgd.npz"
    output_path = figures_dir / "cross_method_summary.pdf"
    _run_if_inputs_exist(
        "cross-method summary",
        [gk_compare_npz, lv_compare_npz],
        output_path,
        lambda: make_cross_method_summary(
            gk_compare_npz=gk_compare_npz,
            lv_compare_npz=lv_compare_npz,
            output_path=output_path,
            figsize=(16, 5),
            dpi=SUMMARY_DPI,
            max_iteration=11000,
        ),
    )


def run_combined_time(root, figures_dir):
    _set_plot_style("combined_time")
    gnk_dir = root / "results" / "gnk"
    lv_dir = root / "results" / "lv"
    gnk_time_paths = [
        gnk_dir / "g_n_k_theta0_3p500_2p000_0p600_m0p800_100.npz",
        gnk_dir / "g_n_k_theta0_3p500_2p000_0p600_m0p800_300.npz",
        gnk_dir / "g_n_k_theta0_3p500_2p000_0p600_m0p800_1000.npz",
        gnk_dir / "g_n_k_theta0_3p500_2p000_0p600_m0p800_3000.npz",
    ]
    lv_time_budget_paths = {
        "50, 60": lv_dir / "lotka_volterra_results_50_60.npz",
        "90, 90": lv_dir / "lotka_volterra_results_90_90.npz",
    }
    lv_corruption_time_paths = {
        "15%": lv_dir / "lotka_volterra_results_60_60_c15.npz",
        "35%": lv_dir / "lotka_volterra_results_60_60_c35.npz",
    }
    output_path = figures_dir / "combined_time_plots.pdf"
    _run_if_inputs_exist(
        "combined time summary",
        [*gnk_time_paths, *lv_time_budget_paths.values(), *lv_corruption_time_paths.values()],
        output_path,
        lambda: make_combined_time_summary(
            gnk_series=collect_gnk_time_series(gnk_time_paths),
            lv_series=collect_lv_time_series(lv_time_budget_paths),
            lv_corruption_series=collect_lv_corruption_time_series(lv_corruption_time_paths),
            output_path=output_path,
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate notebook figures from saved experiment outputs.",
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        default="all",
        choices=(
            "all",
            "mmd_flow",
            "mmd-flow",
            "gnk",
            "gk",
            "g_and_k",
            "lv",
            "gk_ablation",
            "lv_ablation",
            "cross_method",
            "combined_time",
        ),
        help="Experiment to run. Defaults to all.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root containing results/, ablations/, and figures/.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    figures_dir = root / "figures"
    figures_dir.mkdir(exist_ok=True)

    experiment = {
        "mmd-flow": "mmd_flow",
        "gk": "gnk",
        "g_and_k": "gnk",
    }.get(args.experiment, args.experiment)

    if experiment in ("all", "mmd_flow"):
        run_mmd_flow(root, figures_dir)
    if experiment in ("all", "gnk"):
        run_gnk(root, figures_dir)
    if experiment in ("all", "lv"):
        run_lv(root, figures_dir)
    if experiment in ("all", "gk_ablation"):
        run_gnk_ablations(root, figures_dir)
    if experiment in ("all", "lv_ablation"):
        run_lv_ablations(root, figures_dir)
    if experiment in ("all", "cross_method"):
        run_cross_method(root, figures_dir)
    if experiment in ("all", "combined_time"):
        run_combined_time(root, figures_dir)


if __name__ == "__main__":
    main()

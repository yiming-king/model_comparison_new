"""Plot saved RDM comparison tables.

The plotting layer reads only ``results/comparison/n384/<run>/*.csv``. It does
not load neural networks or recompute inference, metrics, thresholds, or
diagnostics.
"""

from pathlib import Path
import argparse
import re

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
import numpy as np
import pandas as pd

from ..config import ASSUMED_MODELS, NUM_TRIALS, PARAM_DIMS, RESULT_DIR
from .comparison import (
    DATASET_KEYS,
    DIAGNOSTICS,
    ERROR_SPECS,
    KEYS,
    SCORING_RULES,
    load_comparison,
)

MODELS = ASSUMED_MODELS
SUMMARY_LABELS = {1: "S=D", 2: "S=2D", 4: "S=4D"}
TITLES = (
    r"$\mathcal{D}_{L_2}$",
    r"$\mathcal{D}_{L_\infty}$",
    r"$\mathcal{D}_{\mathrm{Density}}$",
    r"$\mathcal{D}_{\mathrm{Kernel}}$",
)
XLABELS = (
    r"Diagnostic: $L_2$-based",
    r"Diagnostic: $L_\infty$-based",
    "Diagnostic: density-based",
    "Diagnostic: Kernel-based",
)
PALETTE = ("#0072B2", "#E69F00", "#CC79A7", "#009E73", "#56B4E9", "#D55E00")
SUMMARY_COLORS = {1: PALETTE[0], 2: PALETTE[1], 4: PALETTE[2]}
LOSS_NAMES = {
    "cross_entropy": "Cross-entropy",
    "logistic": "Logistic",
    "exponential": "Exponential",
}

# value column, lower threshold, upper threshold, display conversion, y-label, default scale
METRICS = {
    "posterior_mmd": (
        "posterior_mmd",
        "posterior_error_lower",
        "posterior_error_upper",
        1.0,
        "Posterior MMD\n(NPE, Stan)",
        "linear",
    ),
    "logml": (
        "signed_logml_error",
        "logml_error_lower",
        "logml_error_upper",
        1.0 / np.log(10.0),
        r"$\log_{10}$ marginal-likelihood error" + "\n(NPE - bridge sampling)",
        "symlog",
    ),
    "pmp": (
        "signed_pmp_error",
        "pmp_error_lower",
        "pmp_error_upper",
        1.0,
        r"$\widehat{p}(M_j\mid y)-p(M_j\mid y)$" + "\n(estimated - gold)",
        "linear",
    ),
}

TYPICAL_FILL = "#DCEEDC"
RHO_LINTHRESH = 1.0
RECOVERY_LOSSES = SCORING_RULES


def load_runs(
    method="indirect",
    *,
    summary_multipliers=None,
    scoring_rules=None,
    result_dir=RESULT_DIR,
):
    """Read saved comparison tables and validate cross-run alignment."""
    if method not in ("indirect", "direct"):
        raise ValueError("method must be 'indirect' or 'direct'")
    if method == "indirect" and scoring_rules is not None:
        raise ValueError("scoring_rules selects direct runs only")
    if method == "direct" and summary_multipliers is not None:
        raise ValueError("summary_multipliers selects indirect runs only")

    root = Path(result_dir) / "comparison" / f"n{NUM_TRIALS}"
    pattern = r"indirect_s([124])D" if method == "indirect" else r"direct_(.+)_s12"

    selected = []
    for path in root.glob(f"{method}_*/data.csv"):
        match = re.fullmatch(pattern, path.parent.name)
        if match is None:
            continue
        value = int(match.group(1)) if method == "indirect" else match.group(1)
        choices = summary_multipliers if method == "indirect" else scoring_rules
        if choices is None or value in choices:
            selected.append(value)

    requested = summary_multipliers if method == "indirect" else scoring_rules
    if not selected:
        raise FileNotFoundError(f"No matching {method} comparison tables under {root}")
    if requested is not None and set(requested) - set(selected):
        raise FileNotFoundError(
            f"Missing requested {method} comparisons under {root}: "
            f"{sorted(set(requested) - set(selected), key=str)}"
        )

    if method == "indirect":
        selected.sort()
    else:
        order = {name: i for i, name in enumerate(RECOVERY_LOSSES)}
        selected.sort(key=lambda x: order.get(x, 99))

    runs = {}
    reference_index = None

    for value in selected:
        option = (
            {"summary_multiplier": value}
            if method == "indirect"
            else {"scoring_rule": value}
        )
        frame = load_comparison(method=method, result_dir=result_dir, **option)["data"]

        required = set(KEYS) | {
            "method",
            "num_obs",
            "dataset_type",
            "summary_dim",
            "summary_multiplier",
            "scoring_rule",
            "gold_pmp",
            "estimated_pmp",
            "model_matched",
        }
        numeric = []
        for diagnostic in DIAGNOSTICS:
            numeric.extend((f"rho_{diagnostic}", f"rho_low_{diagnostic}"))

        metric_names = ("pmp",) if method == "direct" else tuple(METRICS)
        for metric in metric_names:
            value_col, low_col, high_col = ERROR_SPECS[metric]
            required.add(value_col)
            required.update((low_col, high_col))
            prefix = "posterior" if metric == "posterior_mmd" else metric
            required.add(f"{prefix}_error_valid")
            numeric.extend((low_col, high_col))

        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{value}: missing comparison columns {sorted(missing)}")

        if frame.empty or frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError(f"{value}: empty data, missing keys, or duplicate rows")
        if not frame.method.eq(method).all() or not frame.num_obs.eq(NUM_TRIALS).all():
            raise ValueError(f"{value}: method/trial metadata disagrees with directory")
        if not set(frame.dataset_type) <= {"simulated", "empirical"}:
            raise ValueError(f"{value}: expected simulated/empirical comparison rows")

        grouped = frame.groupby(DATASET_KEYS, sort=False, dropna=False)
        if set(frame.candidate_model) != set(MODELS) or not grouped.size().eq(len(MODELS)).all():
            raise ValueError(f"{value}: every dataset must contain all four candidates")
        if not np.isfinite(frame[numeric].to_numpy(float)).all():
            raise ValueError(f"{value}: calibrated thresholds/rho values must be finite")

        if method == "indirect":
            expected_dim = frame.candidate_model.map(PARAM_DIMS).to_numpy() * value
            if not frame.summary_multiplier.eq(value).all():
                raise ValueError(f"{value}: summary_multiplier metadata disagrees")
            if not np.array_equal(frame.summary_dim.to_numpy(), expected_dim):
                raise ValueError(f"{value}: summary_dim metadata disagrees")
            label = SUMMARY_LABELS[value]
        else:
            if not frame.scoring_rule.eq(value).all() or not frame.summary_dim.eq(12).all():
                raise ValueError(f"{value}: direct metadata disagrees")
            rho_columns = [f"rho_{diagnostic}" for diagnostic in DIAGNOSTICS]
            if grouped[rho_columns].nunique().gt(1).any().any():
                raise ValueError("Direct diagnostic scores must be shared across candidates")
            label = LOSS_NAMES.get(value, value)

        index = frame.set_index(KEYS).sort_index().index
        if reference_index is not None and not index.equals(reference_index):
            raise ValueError("Dataset/model keys differ across selected runs")
        reference_index = index
        runs[label] = frame

    return runs


def summarize_runs(runs):
    """Return one compact summary row per loaded run."""
    rows = []
    for label, frame in runs.items():
        datasets = frame[DATASET_KEYS].drop_duplicates()
        counts = datasets.dataset_type.value_counts()
        rows.append(
            {
                "run": label,
                "method": str(frame.method.iloc[0]),
                "rows": len(frame),
                "datasets": len(datasets),
                "simulated": int(counts.get("simulated", 0)),
                "empirical": int(counts.get("empirical", 0)),
                "summary_multiplier": int(frame.summary_multiplier.iloc[0]),
                "scoring_rule": str(frame.scoring_rule.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def has_dataset_type(runs, dataset_type):
    return any(dataset_type in set(frame.dataset_type) for frame in runs.values())


def _filter_dataset_type(frame, dataset_type):
    if dataset_type in (None, "both", "all"):
        return frame.copy()
    if dataset_type not in ("simulated", "empirical"):
        raise ValueError("dataset_type must be 'both', 'simulated', or 'empirical'")
    return frame.loc[frame.dataset_type.eq(dataset_type)].copy()


def _constant(frame, column):
    values = frame[column].to_numpy(float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError(f"{column}: no finite values")
    if not np.allclose(values, values[0], rtol=1e-10, atol=1e-12):
        raise ValueError(f"{column} is not constant within the panel")
    return float(values[0])


def _finite(values):
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def _symlog_forward(x, linthresh=RHO_LINTHRESH):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x) / linthresh)


def _symlog_inverse(z, linthresh=RHO_LINTHRESH):
    z = np.asarray(z, dtype=float)
    return np.sign(z) * linthresh * np.expm1(np.abs(z))


def _lowess_curve(x, y, *, frac=0.7, space="symlog", points=120):
    """Local-linear tricube smoother used only for visual guidance."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 4:
        return np.array([]), np.array([])

    if not (0 < frac <= 1):
        raise ValueError("loess_frac must be in (0, 1]")
    if space not in ("raw", "symlog"):
        raise ValueError("loess_space must be 'raw' or 'symlog'")

    tx = _symlog_forward(x) if space == "symlog" else x
    order = np.argsort(tx)
    tx, y = tx[order], y[order]
    k = min(len(tx), max(4, int(np.ceil(frac * len(tx)))))
    grid = np.linspace(tx.min(), tx.max(), min(points, max(20, len(tx))))
    fitted = np.empty_like(grid)

    for i, x0 in enumerate(grid):
        dist = np.abs(tx - x0)
        radius = np.partition(dist, k - 1)[k - 1]
        if radius <= 0:
            mask = dist == 0
            fitted[i] = np.mean(y[mask])
            continue
        u = np.clip(dist / radius, 0, 1)
        weights = (1 - u**3) ** 3
        design = np.column_stack((np.ones(len(tx)), tx - x0))
        weighted = design * np.sqrt(weights)[:, None]
        target = y * np.sqrt(weights)
        beta, *_ = np.linalg.lstsq(weighted, target, rcond=None)
        fitted[i] = beta[0]

    raw_grid = _symlog_inverse(grid) if space == "symlog" else grid
    return raw_grid, fitted


def _set_xscale(ax):
    ax.set_xscale("symlog", linthresh=RHO_LINTHRESH)


def _set_yscale(ax, scale, *, linthresh=0.1):
    if scale == "linear":
        ax.set_yscale("linear")
    elif scale == "symlog":
        ax.set_yscale("symlog", linthresh=linthresh)
    else:
        raise ValueError("yscale must be 'linear' or 'symlog'")


def _limits(axis, values, *, symmetric=False, nonnegative=False):
    values = _finite(values)
    if len(values) == 0:
        return np.array([-1.0, 1.0])
    lo, hi = float(values.min()), float(values.max())
    if np.isclose(lo, hi):
        delta = max(abs(lo) * 0.1, 1e-3)
        lo, hi = lo - delta, hi + delta
    if symmetric:
        bound = max(abs(lo), abs(hi), 1e-6)
        lo, hi = -bound, bound
    transform = axis.get_transform()
    ends = transform.transform([lo, hi])
    padding = 0.04 * max(float(ends[1] - ends[0]), 0.01)
    limits = transform.inverted().transform(ends + [-padding, padding])
    if nonnegative:
        limits[0] = min(0.0, limits[0])
    return limits


def _error_ticks(ax):
    if ax.get_yscale() == "linear":
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        return
    lo, hi = ax.get_ylim()
    transform = ax.yaxis.get_transform()
    a, b = transform.transform([lo, hi])
    ticks_t = np.linspace(a, b, 7)
    ticks = transform.inverted().transform(ticks_t)
    ticks = [float(f"{v:.2g}") for v in ticks if lo <= v <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_locator(NullLocator())


def _save_figure(fig, output, dpi):
    if output is None:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")
        return
    for suffix in (".png", ".pdf"):
        out = path.with_suffix(suffix)
        fig.savefig(out, dpi=dpi if suffix == ".png" else None, bbox_inches="tight")
        print(f"Saved: {out}")


def _facet_strips(fig, axes, *, row_models=True, right_labels=None):
    for ax, title in zip(axes[0], TITLES if axes.shape[1] == 4 else MODELS, strict=True):
        p = ax.get_position()
        strip = fig.add_axes([p.x0, p.y1 + 0.004, p.width, 0.55 / fig.get_figheight()])
        strip.set_facecolor("0.85")
        text = title if axes.shape[1] == 4 else rf"Assumed $M_{{{title[1:]}}}$"
        strip.text(0.5, 0.5, text, ha="center", va="center", fontsize=19)
        strip.set(xticks=[], yticks=[])
        for spine in strip.spines.values():
            spine.set_color("0.35")

    if row_models:
        labels = [rf"Assumed $M_{{{model[1:]}}}$" for model in MODELS]
    else:
        labels = right_labels or []

    for ax, label in zip(axes[:, -1], labels, strict=True):
        p = ax.get_position()
        strip = fig.add_axes([p.x1 + 0.004, p.y0, 0.058, p.height])
        strip.set_facecolor("0.85")
        strip.text(0.5, 0.5, label, rotation=-90, ha="center", va="center", fontsize=17)
        strip.set(xticks=[], yticks=[])
        for spine in strip.spines.values():
            spine.set_color("0.35")


def _run_colors(runs, method):
    colors = {}
    for i, (label, frame) in enumerate(runs.items()):
        if method == "indirect":
            multiplier = int(frame.summary_multiplier.iloc[0])
            colors[label] = SUMMARY_COLORS[multiplier]
        else:
            colors[label] = PALETTE[i % len(PALETTE)]
    return colors


def _scatter_non_pmp(ax, panel, x, y, color):
    simulated = panel.dataset_type.eq("simulated")
    empirical = panel.dataset_type.eq("empirical")
    matched = panel.model_matched.fillna(False).to_numpy(bool)

    masks = (
        (simulated.to_numpy() & matched, "^", 40, 0.30),
        (simulated.to_numpy() & ~matched, "o", 38, 0.30),
        (empirical.to_numpy(), "X", 48, 0.48),
    )
    for mask, marker, size, alpha in masks:
        if mask.any():
            ax.scatter(
                np.asarray(x)[mask],
                np.asarray(y)[mask],
                c=color,
                marker=marker,
                s=size,
                alpha=alpha,
                edgecolors="none",
                zorder=3,
                rasterized=True,
            )

    # Source-level medians make the seven simulated groups readable without
    # replacing the individual observations.
    temp = panel.assign(_x=np.asarray(x), _y=np.asarray(y))
    for (_, source), group in temp.groupby(["dataset_type", "source_model"], sort=False):
        if not np.isfinite(group[["_x", "_y"]].to_numpy(float)).all(axis=1).any():
            continue
        group = group.loc[np.isfinite(group._x) & np.isfinite(group._y)]
        if group.empty:
            continue
        if group.dataset_type.iloc[0] == "empirical":
            marker, size = "X", 115
        elif group.model_matched.fillna(False).all():
            marker, size = "^", 125
        else:
            marker, size = "o", 95
        ax.scatter(
            group._x.median(),
            group._y.median(),
            c=color,
            marker=marker,
            s=size,
            edgecolors="black",
            linewidths=0.65,
            zorder=6,
        )


def _scatter_pmp(ax, panel, x, y, color, *, method, diagnostic):
    if method == "indirect":
        not_all_high = panel[f"at_least_one_not_high_surprise_{diagnostic}"].fillna(False).to_numpy(bool)
        styles = ((~not_all_high, "o", 40), (not_all_high, "D", 58))
    else:
        styles = ((np.ones(len(panel), dtype=bool), "o", 40),)

    simulated = panel.dataset_type.eq("simulated").to_numpy()
    empirical = panel.dataset_type.eq("empirical").to_numpy()

    for mask_class, marker, size in styles:
        sim_mask = mask_class & simulated
        emp_mask = mask_class & empirical
        if sim_mask.any():
            ax.scatter(
                np.asarray(x)[sim_mask],
                np.asarray(y)[sim_mask],
                c=color,
                marker=marker,
                s=size,
                alpha=0.60,
                edgecolors="black",
                linewidths=0.40,
                zorder=3,
                rasterized=True,
            )
        if emp_mask.any():
            ax.scatter(
                np.asarray(x)[emp_mask],
                np.asarray(y)[emp_mask],
                facecolors="none",
                edgecolors=color,
                marker=marker,
                s=size + 12,
                linewidths=1.2,
                zorder=5,
                rasterized=True,
            )


def _metric_legend(fig, runs, colors, *, metric, method, dataset_type,
                   point_color=None, loess_color=None):
    handles = [Line2D([], [], color=colors[label] if loess_color is None else loess_color,
                      lw=3, label=label) for label in runs]
    marker_color = point_color if point_color is not None else ".55"
    marker_edge = point_color if point_color is not None else ".35"

    if metric == "pmp":
        if method == "indirect":
            handles.extend(
                [
                    Line2D([], [], ls="", marker="o", markersize=8, markerfacecolor=marker_color,
                           markeredgecolor="black", label="all high surprise"),
                    Line2D([], [], ls="", marker="D", markersize=8, markerfacecolor=marker_color,
                           markeredgecolor="black", label="at least one not high surprise"),
                ]
            )
            if dataset_type in ("both", "all", None):
                handles.append(
                    Line2D([], [], ls="", marker="o", markersize=9, markerfacecolor="none",
                           markeredgecolor=marker_edge, label="empirical (hollow)")
                )
        else:
            if dataset_type in ("both", "all", None, "simulated"):
                handles.append(
                    Line2D([], [], ls="", marker="o", markersize=8, markerfacecolor=marker_color,
                           markeredgecolor="black", label="simulated (filled)")
                )
            if dataset_type in ("both", "all", None, "empirical"):
                handles.append(
                    Line2D([], [], ls="", marker="o", markersize=9, markerfacecolor="none",
                           markeredgecolor=marker_edge, label="empirical (hollow)")
                )
    else:
        handles.extend(
            [
                Line2D([], [], ls="", marker="^", markersize=8, markerfacecolor=marker_color,
                       markeredgecolor="none", label="well-specified simulated"),
                Line2D([], [], ls="", marker="o", markersize=8, markerfacecolor=marker_color,
                       markeredgecolor="none", label="other simulated"),
                Line2D([], [], ls="", marker="X", markersize=8, markerfacecolor=marker_color,
                       markeredgecolor="none", label="empirical"),
            ]
        )

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(len(handles), 7),
        frameon=False,
        fontsize=14,
        columnspacing=1.1,
        handlelength=1.8,
        handletextpad=0.5,
    )


def plot_metric_grid(
    runs,
    metric="pmp",
    *,
    dataset_type="both",
    output=None,
    ylim=None,
    yscale=None,
    y_linthresh=0.1,
    loess_frac=0.7,
    loess_space="symlog",
    show_loess=True,
    figsize=(21.5, 20),
    dpi=200,
    point_color=None,
    loess_color=None,
):
    """4×4 diagnostic-score versus inference-error grid.

    When ``dataset_type='both'``, each LOESS curve is fitted once to the pooled
    simulated + empirical observations for that run/panel. Dataset type affects
    marker styling only; it never creates separate LOESS fits.
    """
    if not runs or metric not in METRICS:
        raise ValueError("Provide nonempty runs and a valid metric")

    methods = {str(frame.method.iloc[0]) for frame in runs.values()}
    if len(methods) != 1:
        raise ValueError("Plot direct and indirect runs separately")
    method = methods.pop()
    if method == "direct" and metric != "pmp":
        raise ValueError("Direct comparison contains PMP errors only")

    value, lower, upper, factor, ylabel, default_scale = METRICS[metric]
    yscale = yscale or default_scale
    if ylim is not None and (len(ylim) != 2 or ylim[0] >= ylim[1]):
        raise ValueError("ylim must contain two increasing values")

    selected_runs = {
        label: _filter_dataset_type(frame, dataset_type)
        for label, frame in runs.items()
    }
    if any(frame.empty for frame in selected_runs.values()):
        raise ValueError(f"No {dataset_type} rows in one or more selected runs")

    colors = _run_colors(selected_runs, method)
    all_data = pd.concat(selected_runs.values(), ignore_index=True)

    fig, axes = plt.subplots(
        4,
        4,
        figsize=figsize,
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    fig.subplots_adjust(left=0.095, right=0.93, bottom=0.115, top=0.945, wspace=0.10, hspace=0.10)

    for row, model in enumerate(MODELS):
        model_data = all_data.loc[all_data.candidate_model.eq(model)]
        row_y = np.r_[
            model_data[value].to_numpy(float) * factor,
            model_data[lower].to_numpy(float) * factor,
            model_data[upper].to_numpy(float) * factor,
            0.0,
        ]

        for col, diagnostic in enumerate(DIAGNOSTICS):
            ax = axes[row, col]
            xcol = f"rho_{diagnostic}"
            low_x_col = f"rho_low_{diagnostic}"

            panels = {
                label: frame.loc[frame.candidate_model.eq(model)]
                for label, frame in selected_runs.items()
            }
            bounds = {
                label: (_constant(panel, lower) * factor, _constant(panel, upper) * factor)
                for label, panel in panels.items()
            }
            rho_lows = {label: _constant(panel, low_x_col) for label, panel in panels.items()}

            common_x_low = max(rho_lows.values())
            common_y_low = max(bound[0] for bound in bounds.values())
            common_y_high = min(bound[1] for bound in bounds.values())
            if common_x_low <= 1.0:
                ax.axvspan(common_x_low, 1.0, color=TYPICAL_FILL, alpha=0.65, zorder=0)
            if common_y_low <= common_y_high:
                ax.axhspan(common_y_low, common_y_high, color=TYPICAL_FILL, alpha=0.65, zorder=0)

            ax.axvline(1.0, color="0.25", ls="--", lw=0.9, zorder=1)
            if metric != "posterior_mmd":
                ax.axhline(0.0, color="0.35", lw=0.8, zorder=1)

            for label, panel in panels.items():
                color = colors[label]
                lo, hi = bounds[label]
                ax.axvline(rho_lows[label], color=color, ls=":", lw=0.9, alpha=0.55, zorder=1)
                for bound in ((hi,) if metric == "posterior_mmd" else (lo, hi)):
                    ax.axhline(bound, color=color, ls=":", lw=1.15, alpha=0.85, zorder=1)

                x = panel[xcol].to_numpy(float)
                y = panel[value].to_numpy(float) * factor
                valid = np.isfinite(x) & np.isfinite(y)
                if metric == "posterior_mmd":
                    valid &= panel.posterior_error_valid.fillna(False).to_numpy(bool)
                elif metric == "logml":
                    valid &= panel.logml_error_valid.fillna(False).to_numpy(bool)
                else:
                    valid &= panel.pmp_error_valid.fillna(False).to_numpy(bool)

                plot_panel = panel.loc[valid]
                x, y = x[valid], y[valid]
                scatter_color = color if point_color is None else point_color
                if metric == "pmp":
                    _scatter_pmp(ax, plot_panel, x, y, scatter_color, method=method, diagnostic=diagnostic)
                else:
                    _scatter_non_pmp(ax, plot_panel, x, y, scatter_color)

                if show_loess and len(plot_panel) >= 4:
                    curve_x, curve_y = _lowess_curve(
                        x,
                        y,
                        frac=loess_frac,
                        space=loess_space,
                    )
                    if len(curve_x):
                        ax.plot(curve_x, curve_y,
                                color=color if loess_color is None else loess_color,
                                lw=2.0, zorder=7)

            _set_xscale(ax)
            _set_yscale(ax, yscale, linthresh=y_linthresh)

            xvalues = np.r_[all_data[xcol].to_numpy(float), all_data[low_x_col].to_numpy(float), 1.0]
            xlim = _limits(ax.xaxis, xvalues, nonnegative=(diagnostic != "density" and np.nanmin(xvalues) >= 0))
            ax.set_xlim(xlim)

            if ylim is None:
                ylim_auto = _limits(
                    ax.yaxis,
                    row_y,
                    symmetric=(metric == "pmp"),
                    nonnegative=(metric == "posterior_mmd"),
                )
                if metric == "posterior_mmd":
                    ylim_auto[0] = min(-0.01, ylim_auto[0])
                ax.set_ylim(ylim_auto)
            else:
                ax.set_ylim(ylim)

            _error_ticks(ax)
            ax.grid(alpha=0.20)
            ax.tick_params(labelsize=15, labelbottom=(row == 3), labelleft=(col == 0))
            if row == 3:
                ax.set_xlabel(XLABELS[col], fontsize=17)

    fig.supylabel(ylabel, x=0.022, fontsize=21)
    _facet_strips(fig, axes, row_models=True)
    _metric_legend(fig, selected_runs, colors, metric=metric, method=method,
                   dataset_type=dataset_type, point_color=point_color,
                   loess_color=loess_color)
    _save_figure(fig, output, dpi)
    return fig, axes


def plot_pmp_surprise_grid(
    runs,
    *,
    dataset_type="both",
    output=None,
    ylim=(-1.05, 1.05),
    figsize=(21.5, 20),
    dpi=200,
):
    """Indirect PMP error versus the four-model global surprise score.

    The x-axis is min_j rho_j. Therefore x > 1 means all four candidate models
    are high surprise; x <= 1 means at least one candidate is not high surprise.
    No LOESS line is drawn in this classification-focused figure.
    """
    if not runs:
        raise ValueError("runs is empty")
    if {str(frame.method.iloc[0]) for frame in runs.values()} != {"indirect"}:
        raise ValueError("plot_pmp_surprise_grid requires indirect runs")

    selected_runs = {label: _filter_dataset_type(frame, dataset_type) for label, frame in runs.items()}
    colors = _run_colors(selected_runs, "indirect")
    all_global = []

    fig, axes = plt.subplots(4, 4, figsize=figsize, sharex="col", sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.095, right=0.93, bottom=0.115, top=0.945, wspace=0.10, hspace=0.10)

    for row, model in enumerate(MODELS):
        for col, diagnostic in enumerate(DIAGNOSTICS):
            ax = axes[row, col]
            xcol = f"rho_{diagnostic}"
            bounds = {}

            for label, frame in selected_runs.items():
                global_rho = frame.groupby(DATASET_KEYS, sort=False, dropna=False)[xcol].transform("min")
                panel = frame.loc[frame.candidate_model.eq(model)].copy()
                x = global_rho.loc[panel.index].to_numpy(float)
                y = panel.signed_pmp_error.to_numpy(float)
                valid = np.isfinite(x) & np.isfinite(y) & panel.pmp_error_valid.fillna(False).to_numpy(bool)
                panel = panel.loc[valid]
                x, y = x[valid], y[valid]
                all_global.extend(x.tolist())

                lo = _constant(panel, "pmp_error_lower") if len(panel) else np.nan
                hi = _constant(panel, "pmp_error_upper") if len(panel) else np.nan
                bounds[label] = (lo, hi)
                color = colors[label]
                if np.isfinite(lo):
                    ax.axhline(lo, color=color, ls=":", lw=1.1, alpha=0.85)
                if np.isfinite(hi):
                    ax.axhline(hi, color=color, ls=":", lw=1.1, alpha=0.85)

                _scatter_pmp(ax, panel, x, y, color, method="indirect", diagnostic=diagnostic)

            common_low = max(v[0] for v in bounds.values() if np.isfinite(v[0]))
            common_high = min(v[1] for v in bounds.values() if np.isfinite(v[1]))
            if common_low <= common_high:
                ax.axhspan(common_low, common_high, color=TYPICAL_FILL, alpha=0.65, zorder=0)
            ax.axhline(0.0, color="0.35", lw=0.8)
            ax.axvline(1.0, color="0.25", ls="--", lw=0.9)

            _set_xscale(ax)
            ax.set_yscale("linear")
            xvalues = np.asarray(all_global if all_global else [0.0, 1.0])
            xlim = _limits(ax.xaxis, np.r_[xvalues, 1.0])
            ax.set_xlim(xlim)
            ax.axvspan(xlim[0], 1.0, color=TYPICAL_FILL, alpha=0.35, zorder=0)
            ax.set_ylim(ylim)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.grid(alpha=0.20)
            ax.tick_params(labelsize=15, labelbottom=(row == 3), labelleft=(col == 0))
            if row == 3:
                ax.set_xlabel(r"Global surprise score: $\min_j\rho_j$", fontsize=16)

    fig.supylabel(
        r"$\widehat{p}(M_j\mid y)-p(M_j\mid y)$" + "\n(estimated - gold)",
        x=0.022,
        fontsize=21,
    )
    _facet_strips(fig, axes, row_models=True)
    _metric_legend(fig, selected_runs, colors, metric="pmp", method="indirect", dataset_type=dataset_type)
    _save_figure(fig, output, dpi)
    return fig, axes


def plot_gold_pmp_overview(
    runs,
    *,
    summary_multipliers=(1, 4),
    dataset_type="empirical",
    output=None,
    loess_frac=0.7,
    loess_space="symlog",
    figsize=(21.5, 28),
    dpi=200,
):
    """Empirical signed PMP error colored by gold PMP.

    Rows are summary-multiplier × diagnostic combinations; columns are assumed
    models. With the default (1D, 4D), this is an 8×4 grid.
    """
    selected = {}
    for label, frame in runs.items():
        if str(frame.method.iloc[0]) != "indirect":
            continue
        multiplier = int(frame.summary_multiplier.iloc[0])
        if multiplier in summary_multipliers:
            selected[label] = _filter_dataset_type(frame, dataset_type)

    if {int(frame.summary_multiplier.iloc[0]) for frame in selected.values()} != set(summary_multipliers):
        raise ValueError("Requested summary multipliers are not all loaded")

    row_specs = []
    for multiplier in summary_multipliers:
        label = next(label for label, frame in selected.items() if int(frame.summary_multiplier.iloc[0]) == multiplier)
        for diagnostic in DIAGNOSTICS:
            row_specs.append((label, multiplier, diagnostic))

    fig, axes = plt.subplots(len(row_specs), 4, figsize=figsize, sharex=False, sharey=False, squeeze=False)
    fig.subplots_adjust(left=0.095, right=0.89, bottom=0.07, top=0.965, wspace=0.12, hspace=0.14)
    norm = Normalize(vmin=0.0, vmax=1.0)
    last_scatter = None

    for row, (label, multiplier, diagnostic) in enumerate(row_specs):
        frame = selected[label]
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            panel = frame.loc[frame.candidate_model.eq(model)].copy()
            x = panel[f"rho_{diagnostic}"].to_numpy(float)
            y = panel.signed_pmp_error.to_numpy(float)
            c = panel.gold_pmp.to_numpy(float)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(c) & panel.pmp_error_valid.fillna(False).to_numpy(bool)
            panel = panel.loc[valid]
            x, y, c = x[valid], y[valid], c[valid]

            rho_low = _constant(panel, f"rho_low_{diagnostic}")
            err_low = _constant(panel, "pmp_error_lower")
            err_high = _constant(panel, "pmp_error_upper")
            ax.axvspan(rho_low, 1.0, color=TYPICAL_FILL, alpha=0.65, zorder=0)
            ax.axhspan(err_low, err_high, color=TYPICAL_FILL, alpha=0.65, zorder=0)
            ax.axvline(1.0, color="0.25", ls="--", lw=0.9)
            ax.axhline(0.0, color="0.35", lw=0.8)
            ax.axhline(err_low, color="0.3", ls=":", lw=1.0)
            ax.axhline(err_high, color="0.3", ls=":", lw=1.0)

            last_scatter = ax.scatter(
                x,
                y,
                c=c,
                cmap="viridis",
                norm=norm,
                s=58,
                alpha=0.82,
                edgecolors="black",
                linewidths=0.35,
                rasterized=True,
                zorder=3,
            )
            if len(panel) >= 4:
                curve_x, curve_y = _lowess_curve(x, y, frac=loess_frac, space=loess_space)
                if len(curve_x):
                    ax.plot(curve_x, curve_y, color="black", ls="--", lw=1.7, zorder=5)

            _set_xscale(ax)
            ax.set_yscale("linear")
            ax.set_xlim(_limits(ax.xaxis, np.r_[x, rho_low, 1.0]))
            ax.set_ylim(_limits(ax.yaxis, np.r_[y, err_low, err_high, 0.0], symmetric=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.grid(alpha=0.18)
            ax.tick_params(labelsize=12, labelbottom=(row == len(row_specs) - 1), labelleft=(col == 0))
            if row == len(row_specs) - 1:
                ax.set_xlabel(XLABELS[DIAGNOSTICS.index(diagnostic)], fontsize=13)

    # Column strips are assumed models.
    for ax, model in zip(axes[0], MODELS, strict=True):
        p = ax.get_position()
        strip = fig.add_axes([p.x0, p.y1 + 0.004, p.width, 0.45 / fig.get_figheight()])
        strip.set_facecolor("0.85")
        strip.text(0.5, 0.5, rf"Assumed $M_{{{model[1:]}}}$", ha="center", va="center", fontsize=17)
        strip.set(xticks=[], yticks=[])

    right_labels = [f"{SUMMARY_LABELS[multiplier]}\n{TITLES[DIAGNOSTICS.index(diagnostic)]}" for _, multiplier, diagnostic in row_specs]
    for ax, label in zip(axes[:, -1], right_labels, strict=True):
        p = ax.get_position()
        strip = fig.add_axes([p.x1 + 0.004, p.y0, 0.04, p.height])
        strip.set_facecolor("0.85")
        strip.text(0.5, 0.5, label, rotation=-90, ha="center", va="center", fontsize=13)
        strip.set(xticks=[], yticks=[])

    fig.supylabel(
        r"$\widehat{p}(M_j\mid y)-p(M_j\mid y)$" + "\n(estimated - gold)",
        x=0.025,
        fontsize=19,
    )
    if last_scatter is not None:
        cax = fig.add_axes([0.95, 0.18, 0.014, 0.64])
        cbar = fig.colorbar(last_scatter, cax=cax)
        cbar.set_label("Gold-standard PMP", fontsize=15)
        cbar.ax.tick_params(labelsize=12)

    fig.legend(
        handles=[Line2D([], [], color="black", ls="--", lw=1.7, label="LOESS")],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        frameon=False,
        fontsize=14,
    )
    _save_figure(fig, output, dpi)
    return fig, axes


def _select_indirect_run(indirect_runs, summary_multiplier):
    matches = [
        frame
        for frame in indirect_runs.values()
        if int(frame.summary_multiplier.iloc[0]) == summary_multiplier
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one indirect S={summary_multiplier}D run, found {len(matches)}"
        )
    return matches[0]


def _direct_runs_by_loss(direct_runs):
    output = {}
    for frame in direct_runs.values():
        losses = frame.scoring_rule.unique()
        if len(losses) != 1:
            raise ValueError("Each direct run must contain one scoring rule")
        loss = losses[0]
        if loss in output:
            raise ValueError(f"Duplicate direct scoring rule: {loss}")
        output[loss] = frame
    missing = set(RECOVERY_LOSSES) - set(output)
    if missing:
        raise ValueError(f"Missing direct runs: {sorted(missing)}")
    return output


def plot_pmp_recovery(
    indirect_runs,
    direct_runs,
    *,
    summary_multiplier=4,
    dataset_type="both",
    output=None,
    figsize=(23.5, 17.5),
    dpi=200,
):
    """Gold PMP versus indirect/direct estimates for the three direct losses.

    This recovery plot intentionally shows only observations, MAE summaries, and
    the identity line; no LOESS/smoothing curve is fitted.
    """
    indirect = _filter_dataset_type(
        _select_indirect_run(indirect_runs, summary_multiplier),
        dataset_type,
    )
    direct_by_loss = {
        loss: _filter_dataset_type(frame, dataset_type)
        for loss, frame in _direct_runs_by_loss(direct_runs).items()
    }

    indirect_index = indirect.set_index(KEYS).sort_index()
    if not indirect_index.index.is_unique:
        raise ValueError("Indirect PMP rows are not unique")

    fig, axes = plt.subplots(3, 4, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.105, right=0.925, bottom=0.13, top=0.93, wspace=0.12, hspace=0.14)

    for row, loss in enumerate(RECOVERY_LOSSES):
        direct_index = direct_by_loss[loss].set_index(KEYS).sort_index()
        if not indirect_index.index.equals(direct_index.index):
            raise ValueError(f"Indirect/direct keys differ for {loss}")
        if not np.allclose(
            indirect_index.gold_pmp.to_numpy(float),
            direct_index.gold_pmp.to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Gold PMP differs between indirect/direct for {loss}")

        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            mask = indirect_index.index.get_level_values("candidate_model") == model
            gold = indirect_index.loc[mask, "gold_pmp"].to_numpy(float)
            indirect_pmp = indirect_index.loc[mask, "estimated_pmp"].to_numpy(float)
            direct_pmp = direct_index.loc[mask, "estimated_pmp"].to_numpy(float)

            ax.plot([0, 1], [0, 1], "--", color="0.25", lw=1.1, zorder=1)
            ax.scatter(
                gold,
                indirect_pmp,
                s=44,
                marker="o",
                color=PALETTE[0],
                alpha=0.70,
                edgecolors="black",
                linewidths=0.25,
                label="Indirect",
                zorder=2,
            )
            ax.scatter(
                gold,
                direct_pmp,
                s=54,
                marker="^",
                color=PALETTE[1],
                alpha=0.70,
                edgecolors="black",
                linewidths=0.25,
                label="Direct",
                zorder=3,
            )

            mae_indirect = np.mean(np.abs(indirect_pmp - gold))
            mae_direct = np.mean(np.abs(direct_pmp - gold))
            ax.text(
                0.04,
                0.96,
                f"Indirect MAE = {mae_indirect:.3g}\nDirect MAE = {mae_direct:.3g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.4},
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.20)
            ax.tick_params(labelsize=14, labelbottom=(row == 2), labelleft=(col == 0))

    for ax, model in zip(axes[0], MODELS, strict=True):
        p = ax.get_position()
        strip = fig.add_axes([p.x0, p.y1 + 0.004, p.width, 0.50 / fig.get_figheight()])
        strip.set_facecolor("0.85")
        strip.text(0.5, 0.5, rf"$M_{{{model[1:]}}}$", ha="center", va="center", fontsize=19)
        strip.set(xticks=[], yticks=[])

    for ax, loss in zip(axes[:, -1], RECOVERY_LOSSES, strict=True):
        p = ax.get_position()
        strip = fig.add_axes([p.x1 + 0.004, p.y0, 0.06, p.height])
        strip.set_facecolor("0.85")
        strip.text(0.5, 0.5, f"{LOSS_NAMES[loss]} loss", rotation=-90, ha="center", va="center", fontsize=16)
        strip.set(xticks=[], yticks=[])

    fig.supxlabel("Gold-standard PMP", y=0.08, fontsize=18)
    fig.supylabel("Estimated PMP", x=0.025, fontsize=20)
    fig.legend(
        handles=[
            Line2D([], [], ls="", marker="o", markersize=9, markerfacecolor=PALETTE[0], markeredgecolor="black",
                   label=f"Indirect ({SUMMARY_LABELS[summary_multiplier]})"),
            Line2D([], [], ls="", marker="^", markersize=9, markerfacecolor=PALETTE[1], markeredgecolor="black",
                   label="Direct"),
            Line2D([], [], ls="--", color="0.25", lw=1.1, label="Ideal: y=x"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=15,
    )
    _save_figure(fig, output, dpi)
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("indirect", "direct", "both"), default="both")
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--dataset-type", choices=("both", "simulated", "empirical"), default="both")
    parser.add_argument("--summary-multipliers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--scoring-rules", nargs="+", choices=SCORING_RULES, default=list(SCORING_RULES))
    args = parser.parse_args()

    plt.switch_backend("Agg")
    output_root = args.result_dir / "plots" / f"n{NUM_TRIALS}"

    if args.method in ("indirect", "both"):
        indirect = load_runs(
            "indirect",
            summary_multipliers=tuple(args.summary_multipliers),
            result_dir=args.result_dir,
        )
        for metric in METRICS:
            fig, _ = plot_metric_grid(
                indirect,
                metric,
                dataset_type=args.dataset_type,
                output=output_root / "indirect" / args.dataset_type / f"{metric}_4x4",
            )
            plt.close(fig)

    if args.method in ("direct", "both"):
        direct = load_runs(
            "direct",
            scoring_rules=tuple(args.scoring_rules),
            result_dir=args.result_dir,
        )
        fig, _ = plot_metric_grid(
            direct,
            "pmp",
            dataset_type=args.dataset_type,
            output=output_root / "direct" / args.dataset_type / "pmp_4x4",
            ylim=(-1.05, 1.05),
        )
        plt.close(fig)


if __name__ == "__main__":
    main()

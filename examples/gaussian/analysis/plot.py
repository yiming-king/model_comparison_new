from pathlib import Path
import argparse
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
import numpy as np
import pandas as pd

RESULT_DIR = Path(__file__).resolve().parents[1] / "results"
MODELS = ("m1", "m2", "m3", "m4")
DIAGNOSTICS = ("l2", "linf", "density", "mmd")
TITLES = (r"$\mathcal{D}_{L_2}$", r"$\mathcal{D}_{L_\infty}$",
          r"$\mathcal{D}_{\mathrm{Density}}$", r"$\mathcal{D}_{\mathrm{Kernel}}$")
XLABELS = (r"Diagnostic: $L_2$-based", r"Diagnostic: $L_\infty$-based",
           "Diagnostic: density-based", "Diagnostic: Kernel-based")
PALETTE = ("#0072B2", "#E69F00", "#CC79A7", "#009E73", "#56B4E9", "#D55E00")
SUMMARY_COLORS = dict(zip((20, 40, 80), PALETTE))
LOSS_NAMES = {"cross_entropy": "Cross-entropy", "logistic": "Logistic", "exponential": "Exponential"}
KEYS = ["source_model", "id", "candidate_model"]
# column, lower bound, upper bound, conversion from saved units, y-label
METRICS = {
    "posterior_mmd": ("posterior_mmd", "posterior_error_lower", "posterior_error_upper", 1.,
                      "Posterior MMD\n(NPE, analytical)"),
    "logml": ("signed_logml_error", "logml_error_lower", "logml_error_upper", 1. / np.log(10.),
              r"$\log_{10}$ marginal-likelihood error" + "\n(NPE - analytical)"),
    "pmp": ("signed_pmp_error", "pmp_error_lower", "pmp_error_upper", 1.,
            r"$\widehat{p}(M_j\mid y)-p(M_j\mid y)$" + "\n(estimated - analytical)"),
}


def load_runs(method="indirect", num_obs=10, *, summary_dims=None,
              scoring_rules=None, result_dir=RESULT_DIR):
    """Read only data.csv; fail on missing selections or mismatched datasets."""
    if method not in ("indirect", "direct"):
        raise ValueError("method must be 'indirect' or 'direct'")
    root = Path(result_dir) / "comparison" / f"n{num_obs}"
    pattern = r"indirect_s(\d+)" if method == "indirect" else r"direct_(.+)_s(\d+)"
    selected = []
    for path in root.glob(f"{method}_*/data.csv"):
        match = re.fullmatch(pattern, path.parent.name)
        if match is None:
            continue
        dim = int(match.groups()[-1])
        loss = match.group(1) if method == "direct" else None
        if summary_dims is not None and dim not in summary_dims:
            continue
        if scoring_rules is not None and loss not in scoring_rules:
            continue
        selected.append((dim, loss, path))
    if not selected:
        raise FileNotFoundError(f"No matching data.csv under {root}; check copied results and selections.")
    if summary_dims is not None and set(summary_dims) - {s[0] for s in selected}:
        raise FileNotFoundError(f"Missing summary dimensions under {root}: {summary_dims}")
    if scoring_rules is not None and set(scoring_rules) - {s[1] for s in selected}:
        raise FileNotFoundError(f"Missing scoring rules under {root}: {scoring_rules}")
    loss_order = {name: i for i, name in enumerate(LOSS_NAMES)}
    selected.sort(key=lambda s: (s[0], loss_order.get(s[1], 99), s[1] or ""))
    runs, reference = {}, None
    for dim, loss, path in selected:
        frame = pd.read_csv(path)
        required = KEYS + ["method", "num_obs", "summary_dim"]
        numeric = [f"{prefix}{d}" for d in DIAGNOSTICS for prefix in ("rho_", "rho_low_")]
        for metric in (("pmp",) if method == "direct" else METRICS):
            numeric.extend(METRICS[metric][:3])
        missing = set(required + numeric) - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        if frame.empty or frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError(f"{path}: empty data, missing IDs, or duplicate dataset/model rows")
        if not (frame.method.eq(method).all() and frame.num_obs.eq(num_obs).all()
                and frame.summary_dim.eq(dim).all()):
            raise ValueError(f"{path}: metadata disagrees with its directory")
        if loss is not None and ("scoring_rule" not in frame or not frame.scoring_rule.eq(loss).all()):
            raise ValueError(f"{path}: scoring_rule disagrees with its directory")
        groups = frame.groupby(KEYS[:2], dropna=False)
        if set(frame.candidate_model) != set(MODELS) or not groups.size().eq(4).all():
            raise ValueError(f"{path}: every dataset must have exactly M1-M4")
        if not np.isfinite(frame[numeric].to_numpy(float)).all():
            raise ValueError(f"{path}: non-finite errors, rho values, or thresholds")
        if method == "direct" and (groups[[f"rho_{d}" for d in DIAGNOSTICS]].nunique() > 1).any().any():
            raise ValueError(f"{path}: direct rho must be shared across M1-M4")
        index = frame.set_index(KEYS).sort_index().index
        if reference is not None and not index.equals(reference):
            raise ValueError(f"{path}: datasets differ across selected runs")
        reference = index
        label = {20: "S=D", 40: "S=2D", 80: "S=4D"}.get(dim, f"S={dim}")
        if loss is not None:
            label = f"{LOSS_NAMES.get(loss, loss)} (S={dim})"
        runs[label] = frame
    return runs


def _constant(frame, column):
    values = frame[column].to_numpy(float)
    if not np.allclose(values, values[0], rtol=1e-10, atol=1e-12):
        raise ValueError(f"{column} is not constant within a candidate-model panel")
    return float(values[0])


def _limits(axis, values, *, symmetric=False, nonnegative=False):
    values = np.asarray(values, dtype=float).ravel()
    lo, hi = float(values.min()), float(values.max())
    if symmetric:
        hi = max(abs(lo), abs(hi), 1e-6)
        lo = -hi
    transform = axis.get_transform()
    ends = transform.transform([lo, hi])
    padding = .04 * max(float(ends[1] - ends[0]), .01)
    limits = transform.inverted().transform(ends + [-padding, padding])
    if nonnegative:
        limits[0] = 0.
    return limits


def _scatter(ax, frame, x, y, color, metric, method, flags):
    if metric == "pmp":
        # The indirect classification uses ALL four candidates, not this row only.
        styles = ((False, "o", 40), (True, "D", 60)) if method == "indirect" else ((False, "o", 40),)
        for flag, marker, size in styles:
            mask = flags.eq(flag) if method == "indirect" else np.ones(len(frame), dtype=bool)
            ax.scatter(x[mask], y[mask], c=color, marker=marker, s=size, alpha=.62,
                       edgecolors="black", linewidths=.45, zorder=3, rasterized=True)
        return
    well = frame.source_model.eq(frame.candidate_model)
    for flag, marker in ((True, "^"), (False, "o")):
        mask = well.eq(flag)
        ax.scatter(x[mask], y[mask], c=color, marker=marker, s=42, alpha=.28,
                   edgecolors="none", zorder=3, rasterized=True)
    for _, group in frame.assign(_x=x, _y=y).groupby("source_model", sort=False):
        is_well = group.source_model.eq(group.candidate_model).all()
        ax.scatter(group._x.median(), group._y.median(), c=color, marker="^" if is_well else "o",
                   s=132 if is_well else 96, edgecolors="black", linewidths=.65, zorder=6)


def _error_ticks(ax, metric):
    if ax.get_yscale() == "linear":
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        return
    lo, hi = ax.get_ylim()
    if metric == "posterior_mmd":
        ticks = [0.] + [10. ** e for e in range(-1, int(np.floor(np.log10(max(hi, .1)))) + 1)]
        ticks = [v for v in ticks if lo <= v <= hi]
    else:
        transform = ax.yaxis.get_transform()
        a, b = transform.transform([lo, hi])
        step = (b - a) / 6
        indices = np.arange(np.ceil(a / step), np.floor(b / step) + 1)
        ticks = sorted({float(f"{v:.1g}") for v in transform.inverted().transform(indices * step)})
        ticks = [v for v in ticks if lo <= v <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())


def _strips(fig, axes):
    positions = []
    for ax, title in zip(axes[0], TITLES):
        p = ax.get_position()
        positions.append(([p.x0, p.y1 + .004, p.width, .55 / fig.get_figheight()], title, 0))
    for ax, model in zip(axes[:, -1], MODELS):
        p = ax.get_position()
        positions.append(([p.x1 + .004, p.y0, .058, p.height], rf"Assumed $M_{{{model[1:]}}}$", -90))
    for position, text, rotation in positions:
        strip = fig.add_axes(position, facecolor="0.85")
        strip.text(.5, .5, text, ha="center", va="center", rotation=rotation, fontsize=19)
        strip.set(xticks=[], yticks=[])
        for spine in strip.spines.values():
            spine.set_color("0.35")


def plot_metric_grid(runs, metric="pmp", *, output=None, ylim=None,
                     yscale=None, figsize=(21.5, 20), dpi=200):
    """Return (figure, axes). Input must contain one method; CSVs are never modified."""
    if not runs or metric not in METRICS:
        raise ValueError("Provide nonempty runs and metric='posterior_mmd', 'logml', or 'pmp'")
    methods = {str(frame.method.iloc[0]) for frame in runs.values()}
    if len(methods) != 1:
        raise ValueError("Plot direct and indirect separately; use the same PMP ylim to compare")
    method = methods.pop()
    if method == "direct" and metric != "pmp":
        raise ValueError("Direct comparison only provides PMP errors")
    value, lower, upper, factor, ylabel = METRICS[metric]
    if ylim is not None and (len(ylim) != 2 or not np.isfinite(ylim).all() or ylim[0] >= ylim[1]):
        raise ValueError("ylim must contain two finite increasing values")
    colors = {label: (SUMMARY_COLORS.get(int(f.summary_dim.iloc[0]), PALETTE[i % len(PALETTE)])
                      if method == "indirect" else PALETTE[i % len(PALETTE)])
              for i, (label, f) in enumerate(runs.items())}
    yscale = yscale or ("linear" if metric == "pmp" else "symlog")
    if yscale not in ("linear", "symlog"):
        raise ValueError("yscale must be 'linear' or 'symlog'")
    fig, axes = plt.subplots(4, 4, figsize=figsize, sharex="col",
                             sharey=True if metric == "pmp" else "row", squeeze=False)
    fig.subplots_adjust(left=.095, right=.93, bottom=.105, top=.945, wspace=.10, hspace=.10)
    all_data = pd.concat(runs.values(), ignore_index=True)
    for row, model in enumerate(MODELS):
        ydata = all_data if metric == "pmp" else all_data.loc[all_data.candidate_model.eq(model)]
        yvalues = ydata[[value, lower, upper]].to_numpy(float).ravel() * factor
        for col, diagnostic in enumerate(DIAGNOSTICS):
            ax = axes[row, col]
            xcol, lowcol = f"rho_{diagnostic}", f"rho_low_{diagnostic}"
            panels = {label: f.loc[f.candidate_model.eq(model)] for label, f in runs.items()}
            bounds = {label: (_constant(p, lower) * factor, _constant(p, upper) * factor)
                      for label, p in panels.items()}
            rho_lows = [_constant(p, lowcol) for p in panels.values()]
            if any(lo > hi for lo, hi in bounds.values()) or max(rho_lows) > 1. + 1e-10:
                raise ValueError("Invalid calibrated interval")
            # Same x-envelope as the old notebook. Actual classification never uses this band.
            ax.axvspan(min(rho_lows), 1., color="#DCEEDC", alpha=.70, zorder=0)
            ax.axvline(1., color="0.25", ls="--", lw=.9, zorder=1)
            lo, hi = max(b[0] for b in bounds.values()), min(b[1] for b in bounds.values())
            if lo <= hi:  # No green 'acceptable' band when the intersection is empty.
                ax.axhspan(lo, hi, color="#DCEEDC", alpha=.70, zorder=0)
            for label, panel in panels.items():
                lo, hi = bounds[label]
                for bound in ((hi,) if metric == "posterior_mmd" else (lo, hi)):
                    ax.axhline(bound, color=colors[label], ls=":", lw=1.2, alpha=.9, zorder=1)
                flags = None
                if metric == "pmp" and method == "indirect":
                    flags = runs[label].groupby(KEYS[:2])[xcol].transform("min").le(1.).loc[panel.index]
                _scatter(ax, panel, panel[xcol], panel[value] * factor, colors[label], metric, method, flags)
            if metric != "posterior_mmd":
                ax.axhline(0., color="0.35", lw=.8, zorder=1)
            ax.set_xscale("symlog", linthresh=1.)
            linscale = {"posterior_mmd": .2, "logml": 3., "pmp": 1.}[metric]
            ax.set_yscale(yscale, **({"linthresh": .1, "linscale": linscale}
                                    if yscale == "symlog" else {}))
            xvalues = np.r_[all_data[xcol], all_data[lowcol], 1.]
            xlimits = _limits(ax.xaxis, xvalues,
                              nonnegative=diagnostic != "mmd" and xvalues.min() >= 0.)
            if diagnostic == "mmd" and xvalues.min() >= 0.:
                xlimits[0] = max(0., xlimits[0])
            ax.set_xlim(xlimits)
            limits = _limits(ax.yaxis, np.r_[yvalues, 0.], symmetric=(metric == "pmp" or (metric == "logml" and model == "m3")))
            if metric == "posterior_mmd":
                limits[0] = -.01
            ax.set_ylim(limits if ylim is None else ylim)
            _error_ticks(ax, metric)
            if diagnostic == "mmd" and np.max(xvalues) <= 2.:
                ax.xaxis.set_major_locator(FixedLocator([.5, 1., 1.5, 2.]))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(alpha=.2)
            ax.tick_params(labelsize=17, labelbottom=row == 3, labelleft=col == 0)
            if row == 3:
                ax.set_xlabel(XLABELS[col], fontsize=19)
    fig.supylabel(ylabel, x=.022, fontsize=22)
    _strips(fig, axes)
    handles = [Line2D([], [], color=color, lw=3, label=label) for label, color in colors.items()]
    shapes = [("o", "simulated datasets", False)]
    if metric != "pmp":
        shapes = [("^", "well-specified datasets", False), ("o", "other simulated datasets", False),
                  ("^", "well-specified median", True), ("o", "other simulated median", True)]
    elif method == "indirect":
        shapes = [("o", "all high surprise", True), ("D", "at least one not high surprise", True)]
    for marker, label, edged in shapes:
        handles.append(Line2D([], [], ls="", marker=marker, markersize=9, markerfacecolor=".55",
                              markeredgecolor="black" if edged else "none", label=label))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .3125 / fig.get_figheight()),
               ncol=min(len(handles), 7), frameon=False, fontsize=16,
               columnspacing=1.15, handlelength=1.8, handletextpad=.5)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output}")
    return fig, axes


RECOVERY_LOSSES = ("cross_entropy", "exponential", "logistic")
RECOVERY_METHOD_COLORS = {"indirect": "#0072B2", "direct": "#E69F00"}
RECOVERY_METHOD_MARKERS = {"indirect": "o", "direct": "^"}
RECOVERY_SOURCE_COLORS = (
    "#F0E442", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#0072B2", "#E011CF",
    "#999999", "#1F03EE", "#882255", "#44AA99",
)


def load_pmp_recovery_data(indirect_runs, direct_runs, *, summary_dim=80):
    """Align S=4D indirect, three direct losses, and gold PMP by dataset/model."""
    selected = [frame for frame in indirect_runs.values()
                if int(frame.summary_dim.iloc[0]) == summary_dim]
    if len(selected) != 1:
        raise ValueError(f"Expected one indirect S={summary_dim} run, found {len(selected)}")
    indirect = selected[0].set_index(KEYS).sort_index()
    if not indirect.index.is_unique:
        raise ValueError("Indirect PMP has duplicate dataset/model keys")
    direct_by_loss = {}
    for frame in direct_runs.values():
        losses = frame.scoring_rule.unique()
        if len(losses) != 1 or losses[0] in direct_by_loss:
            raise ValueError("Each direct run must have one distinct scoring rule")
        direct_by_loss[losses[0]] = frame
    if set(direct_by_loss) != set(RECOVERY_LOSSES):
        raise ValueError(f"Direct runs must contain {RECOVERY_LOSSES}")

    frames = []
    for loss in RECOVERY_LOSSES:
        direct = direct_by_loss[loss].set_index(KEYS).sort_index()
        if not direct.index.is_unique or not indirect.index.equals(direct.index):
            raise ValueError(f"Indirect/direct dataset keys differ for {loss}")
        if not direct.scoring_rule.eq(loss).all():
            raise ValueError(f"Direct scoring rule differs from {loss}")
        if not np.array_equal(indirect.gold_pmp.to_numpy(), direct.gold_pmp.to_numpy()):
            raise ValueError(f"Gold PMP differs for {loss}")
        frame = pd.DataFrame({
            "gold_pmp": indirect.gold_pmp,
            "indirect_pmp": indirect.estimated_pmp,
            "direct_pmp": direct.estimated_pmp,
        }).reset_index()
        frame["scoring_rule"] = loss
        frame["summary_dim"] = summary_dim
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    values = data[["gold_pmp", "indirect_pmp", "direct_pmp"]].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("PMP values must be finite and in [0, 1]")
    return data


def _recovery_facet_strips(fig, axes):
    """Use the same facet strips as the reference PMP comparison notebook."""
    for ax, model in zip(axes[0], MODELS, strict=True):
        pos = ax.get_position()
        strip = fig.add_axes([pos.x0, pos.y1 + .004, pos.width,
                              .55 / fig.get_figheight()])
        strip.set_facecolor("0.85")
        strip.text(.5, .5, rf"$M_{{{model[1:]}}}$", ha="center", va="center", fontsize=21)
        strip.set(xticks=[], yticks=[])
        for spine in strip.spines.values():
            spine.set_color("0.35")
    for ax, loss in zip(axes[:, -1], RECOVERY_LOSSES, strict=True):
        pos = ax.get_position()
        strip = fig.add_axes([pos.x1 + .004, pos.y0, .058, pos.height])
        strip.set_facecolor("0.85")
        strip.text(.5, .5, f"{LOSS_NAMES[loss]} loss", rotation=-90,
                   ha="center", va="center", fontsize=19)
        strip.set(xticks=[], yticks=[])
        for spine in strip.spines.values():
            spine.set_color("0.35")


def _save_recovery_figure(fig, output_stem, dpi):
    if output_stem is None:
        return
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=dpi if suffix == ".png" else None,
                    bbox_inches="tight")
        print(f"Saved: {path}")


def _validate_recovery_data(data):
    required = set(KEYS) | {"scoring_rule", "summary_dim", "gold_pmp",
                            "indirect_pmp", "direct_pmp"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing PMP recovery columns: {sorted(missing)}")
    if set(data.scoring_rule) != set(RECOVERY_LOSSES):
        raise ValueError(f"Expected direct losses {RECOVERY_LOSSES}")
    if set(data.candidate_model) != set(MODELS):
        raise ValueError(f"Expected candidate models {MODELS}")
    if data.duplicated(["scoring_rule", *KEYS]).any():
        raise ValueError("Duplicate PMP recovery rows")
    if data.summary_dim.nunique() != 1:
        raise ValueError("PMP recovery plots require one indirect summary dimension")


def _recovery_indirect_label(data):
    dim = int(data.summary_dim.iloc[0])
    label = {20: "D", 40: "2D", 80: "4D"}.get(dim, str(dim))
    return f"Indirect PMP (S={label})"


def plot_pmp_recovery(data, *, output_stem=None, dpi=200):
    """Plot gold versus indirect/direct PMP, colored by method (reference figure 1)."""
    _validate_recovery_data(data)
    with plt.rc_context({"font.size": 17, "savefig.facecolor": "white"}):
        figure_height = 17.5
        fig, axes = plt.subplots(3, 4, figsize=(23.5, figure_height),
                                 sharex=True, sharey=True)
        fig.subplots_adjust(left=.105, right=.925, bottom=2.4 / figure_height,
                            top=1. - 1.2 / figure_height, wspace=.12, hspace=.14)
        for loss, row_axes in zip(RECOVERY_LOSSES, axes, strict=True):
            run = data.loc[data.scoring_rule.eq(loss)]
            for model, ax in zip(MODELS, row_axes, strict=True):
                panel = run.loc[run.candidate_model.eq(model)]
                gold = panel.gold_pmp.to_numpy(float)
                indirect = panel.indirect_pmp.to_numpy(float)
                direct = panel.direct_pmp.to_numpy(float)
                ax.set_axisbelow(True)
                ax.grid(alpha=.2)
                ax.plot([0, 1], [0, 1], "--", color="0.25", linewidth=1.2, zorder=1)
                for method, estimate in (("indirect", indirect), ("direct", direct)):
                    ax.scatter(gold, estimate, s=45, marker="o",
                               color=RECOVERY_METHOD_COLORS[method], alpha=.75,
                               edgecolors="black", linewidths=.25,
                               zorder=2 if method == "indirect" else 3,
                               clip_on=False)
                for y, method, estimate in ((.95, "indirect", indirect),
                                            (.86, "direct", direct)):
                    mae = np.mean(np.abs(estimate - gold))
                    ax.text(.04, y, f"{method.capitalize()} MAE = {mae:.4g}",
                            transform=ax.transAxes, ha="left", va="top",
                            color=RECOVERY_METHOD_COLORS[method], fontsize=13,
                            bbox={"facecolor": "white", "edgecolor": "none",
                                  "alpha": .8, "pad": 1.4}, zorder=4)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks(np.linspace(0, 1, 6))
                ax.set_yticks(np.linspace(0, 1, 6))
                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(labelbottom=loss == RECOVERY_LOSSES[-1],
                               labelleft=model == MODELS[0], labelsize=17)
        _recovery_facet_strips(fig, axes)
        fig.supxlabel("Gold-standard PMP", y=.105, fontsize=19)
        fig.supylabel("Estimated PMP", x=.025, fontsize=22)
        handles = [
            Line2D([], [], linestyle="none", marker="o", markersize=11,
                   markerfacecolor=RECOVERY_METHOD_COLORS[method],
                   markeredgecolor="black", markeredgewidth=.3, label=label)
            for method, label in (("indirect", _recovery_indirect_label(data)),
                                  ("direct", "Direct PMP"))
        ]
        handles.append(Line2D([], [], linestyle="--", color="0.25",
                              linewidth=1.2, label="Ideal: y = x"))
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(.5, .3125 / figure_height), ncol=3,
                   frameon=False, fontsize=18, columnspacing=2.)
        _save_recovery_figure(fig, output_stem, dpi)
    return fig, axes


def plot_pmp_recovery_by_source(data, *, output_stem=None, dpi=200):
    """Plot gold versus indirect/direct PMP, colored by source (reference figure 2)."""
    _validate_recovery_data(data)
    sources = sorted(data.source_model.unique(),
                     key=lambda source: int(str(source).lower().removeprefix("m")))
    if len(sources) > len(RECOVERY_SOURCE_COLORS):
        raise ValueError("More source models than available reference colors")
    source_colors = dict(zip(sources, RECOVERY_SOURCE_COLORS, strict=True))
    with plt.rc_context({"font.size": 17, "savefig.facecolor": "white"}):
        figure_height = 17.5
        fig, axes = plt.subplots(3, 4, figsize=(23.5, figure_height),
                                 sharex=True, sharey=True)
        fig.subplots_adjust(left=.105, right=.925, bottom=3.1 / figure_height,
                            top=1. - 1.2 / figure_height, wspace=.12, hspace=.14)
        for loss, row_axes in zip(RECOVERY_LOSSES, axes, strict=True):
            run = data.loc[data.scoring_rule.eq(loss)]
            for model, ax in zip(MODELS, row_axes, strict=True):
                panel = run.loc[run.candidate_model.eq(model)]
                ax.set_axisbelow(True)
                ax.grid(alpha=.2)
                ax.plot([0, 1], [0, 1], "--", color="0.25", linewidth=1.2, zorder=1)
                for source in sources:
                    sub = panel.loc[panel.source_model.eq(source)]
                    gold = sub.gold_pmp.to_numpy(float)
                    for method, column in (("indirect", "indirect_pmp"),
                                           ("direct", "direct_pmp")):
                        ax.scatter(gold, sub[column].to_numpy(float),
                                   s=48 if method == "indirect" else 55,
                                   marker=RECOVERY_METHOD_MARKERS[method],
                                   color=source_colors[source], alpha=.75,
                                   edgecolors="black", linewidths=.2,
                                   zorder=2 if method == "indirect" else 3,
                                   clip_on=False)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks(np.linspace(0, 1, 6))
                ax.set_yticks(np.linspace(0, 1, 6))
                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(labelbottom=loss == RECOVERY_LOSSES[-1],
                               labelleft=model == MODELS[0], labelsize=17)
        _recovery_facet_strips(fig, axes)
        fig.supxlabel("Gold-standard PMP", y=.145, fontsize=19)
        fig.supylabel("Estimated PMP", x=.025, fontsize=22)
        method_handles = [
            Line2D([], [], linestyle="none", marker=RECOVERY_METHOD_MARKERS[method],
                   markersize=11, markerfacecolor="0.35", markeredgecolor="black",
                   markeredgewidth=.3, label=label)
            for method, label in (("indirect", _recovery_indirect_label(data)),
                                  ("direct", "Direct PMP"))
        ]
        method_handles.append(Line2D([], [], linestyle="--", color="0.25",
                                     linewidth=1.2, label="Ideal: y = x"))
        source_handles = [
            Line2D([], [], linestyle="none", marker="o", markersize=10,
                   markerfacecolor=source_colors[source], markeredgecolor="black",
                   markeredgewidth=.3, label=str(source).upper())
            for source in sources
        ]
        fig.legend(handles=method_handles, loc="lower center",
                   bbox_to_anchor=(.5, 1.5 / figure_height), ncol=3,
                   frameon=False, fontsize=18, columnspacing=2.)
        fig.legend(handles=source_handles, loc="lower center",
                   bbox_to_anchor=(.5, .1 / figure_height),
                   ncol=min(6, len(sources)), frameon=False, fontsize=16,
                   columnspacing=1.8, handletextpad=.5, labelspacing=.8)
        _save_recovery_figure(fig, output_stem, dpi)
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--method", choices=("indirect", "direct", "both"), default="both")
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--indirect-dims", nargs="+", type=int)
    parser.add_argument("--direct-dims", nargs="+", type=int)
    parser.add_argument("--losses", nargs="+")
    args = parser.parse_args()
    plt.switch_backend("Agg")
    methods = ("indirect", "direct") if args.method == "both" else (args.method,)
    for method in methods:
        dims = args.indirect_dims if method == "indirect" else args.direct_dims
        runs = load_runs(method, args.n, summary_dims=dims, result_dir=args.result_dir,
                         scoring_rules=args.losses if method == "direct" else None)
        print(f"{method}: {', '.join(runs)}")
        for metric in (METRICS if method == "indirect" else ("pmp",)):
            path = args.result_dir / "plots" / f"n{args.n}" / method / f"{metric}_4x4.png"
            fig, _ = plot_metric_grid(runs, metric, output=path,
                                      ylim=(-1.05, 1.05) if metric == "pmp" else None)
            plt.close(fig)


if __name__ == "__main__":
    main()

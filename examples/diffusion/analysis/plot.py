"""Plot saved RDM comparison tables in the Gaussian/reference-notebook style.

Only NumPy, Pandas, SciPy, and Matplotlib are needed. LOESS is a local-linear
tricube regression, implemented here so notebooks contain only calls.
"""

from pathlib import Path
import argparse
import re

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator, SymmetricalLogLocator
import numpy as np
import pandas as pd

from ..config import ASSUMED_MODELS, NUM_TRIALS, PARAM_DIMS, RESULT_DIR
from .comparison import DATASET_KEYS, DIAGNOSTICS, ERROR_SPECS, KEYS, load_comparison

MODELS = ASSUMED_MODELS
TITLES = (r"$L_2$", r"$L_\infty$", "Density", "Kernel")
XLABELS = (r"Diagnostic: $L_2$-based", r"Diagnostic: $L_\infty$-based",
           "Diagnostic: density-based", "Diagnostic: Kernel-based")
PALETTE = ("#0072B2", "#E69F00", "#CC79A7", "#009E73", "#56B4E9", "#D55E00")
SUMMARY_COLORS = dict(zip((1, 2, 4), PALETTE[:3]))
LOSS_NAMES = {"cross_entropy": "Cross-entropy", "logistic": "Logistic", "exponential": "Exponential"}
METRICS = {
    "posterior_mmd": (1., "Posterior MMD\n(NPE, Stan)", "linear", 1.),
    "logml": (1. / np.log(10.), r"$\log_{10}$ marginal-likelihood error" + "\n(NPE - bridge sampling)", "symlog", 1.),
    "pmp": (1., r"$\widehat{p}(M_j\mid y)-p(M_j\mid y)$" + "\n(estimated - gold)", "linear", .1),
}
METRIC_ALIASES = {"log10_logml_error": "logml", "pmp_error": "pmp"}


def load_runs(method="indirect", *, summary_multipliers=None, scoring_rules=None, result_dir=RESULT_DIR):
    """Read saved comparisons and require identical dataset/model keys per method."""
    if method not in ("indirect", "direct"):
        raise ValueError("method must be indirect or direct")
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
        value = int(match[1]) if method == "indirect" else match[1]
        choices = summary_multipliers if method == "indirect" else scoring_rules
        if choices is None or value in choices:
            selected.append(value)
    requested = summary_multipliers if method == "indirect" else scoring_rules
    if not selected or (requested is not None and set(requested) - set(selected)):
        raise FileNotFoundError(f"Missing requested {method} comparisons under {root}: {requested}")
    selected.sort(key=lambda x: x if method == "indirect" else list(LOSS_NAMES).index(x))
    runs, reference = {}, None
    for value in selected:
        option = {"summary_multiplier": value} if method == "indirect" else {"scoring_rule": value}
        frame = load_comparison(method=method, result_dir=result_dir, **option)["data"]
        required = set(KEYS) | {"method", "num_obs", "summary_dim", "summary_multiplier", "scoring_rule", "gold_pmp", "estimated_pmp", "model_matched"}
        numeric = [f"{prefix}{d}" for d in DIAGNOSTICS for prefix in ("rho_", "rho_low_")]
        for metric in (("pmp",) if method == "direct" else METRICS):
            numeric.extend(ERROR_SPECS[metric][1:])
            prefix = "posterior" if metric == "posterior_mmd" else metric
            required.update((ERROR_SPECS[metric][0], f"{prefix}_error_valid"))
        missing = (required | set(numeric)) - set(frame)
        if missing:
            raise ValueError(f"{value}: missing comparison columns {sorted(missing)}")
        if frame.empty or frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError(f"{value}: empty data, missing IDs, or duplicate rows")
        if not frame.method.eq(method).all() or not frame.num_obs.eq(NUM_TRIALS).all():
            raise ValueError(f"{value}: invalid method or trial count")
        if not set(frame.dataset_type) <= {"simulated", "empirical"}:
            raise ValueError(f"{value}: dataset_type must be simulated or empirical")
        grouped = frame.groupby(DATASET_KEYS, sort=False, dropna=False)
        if set(frame.candidate_model) != set(MODELS) or not grouped.size().eq(len(MODELS)).all():
            raise ValueError(f"{value}: each dataset must have all four candidates")
        if not np.isfinite(frame[numeric].to_numpy(float)).all():
            raise ValueError(f"{value}: rho and calibrated thresholds must be finite")
        if method == "indirect":
            expected = frame.candidate_model.map(PARAM_DIMS).to_numpy() * value
            if not frame.summary_multiplier.eq(value).all() or not np.array_equal(frame.summary_dim, expected):
                raise ValueError(f"{value}: summary metadata disagrees with its directory")
        else:
            if not frame.scoring_rule.eq(value).all() or not frame.summary_dim.eq(12).all():
                raise ValueError(f"{value}: direct metadata disagrees with its directory")
            if grouped[[f"rho_{d}" for d in DIAGNOSTICS]].nunique().gt(1).any().any():
                raise ValueError("Direct diagnostic scores must be shared across candidates")
        index = frame.set_index(KEYS).sort_index().index
        if reference is not None and not index.equals(reference):
            raise ValueError("Dataset/model keys differ across selected runs")
        reference = index
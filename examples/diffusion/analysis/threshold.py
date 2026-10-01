"""Calibrate diffusion inference-error thresholds from benchmark datasets."""

from pathlib import Path

import numpy as np

from ..config import ASSUMED_MODELS, RESULT_DIR
from .inference import DIRECT_SUMMARY_DIM
from .metric import load_npz, metrics_path

THRESHOLD_DIR = RESULT_DIR / "thresholds"


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------


def central_interval(values, *, coverage=0.90, axis=0):
    """Return a central empirical interval."""
    values = np.asarray(values, dtype=np.float64)
    alpha = 1.0 - coverage
    lower = np.quantile(values, alpha / 2.0, axis=axis)
    upper = np.quantile(values, 1.0 - alpha / 2.0, axis=axis)
    return lower, upper

def upper_quantile(values, *, quantile=0.95, axis=0):
    """Return an upper empirical quantile."""
    values = np.asarray(values, dtype=np.float64)
    return np.quantile(values, quantile, axis=axis)

def save_thresholds(thresholds, path, *, overwrite=False):
    """Save calibrated thresholds."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Threshold file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **thresholds)

def load_thresholds(path):
    """Load calibrated thresholds."""
    return load_npz(path)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------


def threshold_path(*, method, summary_multiplier=None, scoring_rule=None):
    """Return the path for one threshold file."""
    if method == "indirect":
        if summary_multiplier is None:
            raise ValueError("summary_multiplier is required for indirect thresholds")
        name = f"indirect_s{summary_multiplier}D.npz"
    elif method == "direct":
        if scoring_rule is None:
            raise ValueError("scoring_rule is required for direct thresholds")
        name = f"direct_{scoring_rule}_s{DIRECT_SUMMARY_DIM}.npz"
    else:
        raise ValueError("method must be 'indirect' or 'direct'")
    return THRESHOLD_DIR / "n384" / name


# ---------------------------------------------------------------------
# Load benchmark metrics
# ---------------------------------------------------------------------


def load_all_metrics(*, method, summary_multiplier=None, scoring_rule=None):
    """
    Load benchmark metrics generated from M0-M3.

    Returns:
        {
            "m0": metrics,
            "m1": metrics,
            "m2": metrics,
            "m3": metrics,
        }
    """
    results = {}
    for source in ASSUMED_MODELS:
        path = metrics_path(
            method=method,
            split="benchmark",
            source=source,
            summary_multiplier=summary_multiplier,
            scoring_rule=scoring_rule,
        )
        results[source] = load_npz(path)
    return results


# ---------------------------------------------------------------------
# Indirect thresholds
# ---------------------------------------------------------------------


def calibrate_indirect_thresholds(metrics_by_source, *, posterior_quantile=0.95, signed_error_coverage=0.90):
    """
    Calibrate indirect inference-error thresholds.

    Posterior MMD:
        matching generating/candidate model
        converged Stan fit only
        [0, q95]

    logML:
        matching generating/candidate model
        converged Stan fit only
        central 90%

    PMP:
        combine benchmark datasets from M0-M3
        require all candidate Stan fits to converge
        central 90% for each PMP component
    """
    num_models = len(ASSUMED_MODELS)
    posterior_mmd_upper = np.empty(num_models, dtype=np.float64)
    logml_error_lower = np.empty(num_models, dtype=np.float64)
    logml_error_upper = np.empty(num_models, dtype=np.float64)
    posterior_counts = np.empty(num_models, dtype=np.int64)
    logml_counts = np.empty(num_models, dtype=np.int64)
    # --------------------------------------------------
    # Posterior MMD + logML
    # Matching generating model and candidate model
    # --------------------------------------------------
    for model_index, model in enumerate(ASSUMED_MODELS):
        metrics = metrics_by_source[model]
        valid = np.asarray(metrics["converged"][:, model_index], dtype=bool)
        posterior_values = np.asarray(metrics["posterior_mmd"][:, model_index], dtype=np.float64)
        logml_values = np.asarray(metrics["signed_logml_error"][:, model_index], dtype=np.float64)
        posterior_valid = valid & np.isfinite(posterior_values)
        logml_valid = valid & np.isfinite(logml_values)
        posterior_values = posterior_values[posterior_valid]
        logml_values = logml_values[logml_valid]
        if len(posterior_values) == 0:
            raise ValueError(f"No valid posterior MMD values for {model}")
        if len(logml_values) == 0:
            raise ValueError(f"No valid logML errors for {model}")
        posterior_mmd_upper[model_index] = upper_quantile(posterior_values, quantile=posterior_quantile)
        (logml_error_lower[model_index], logml_error_upper[model_index]) = central_interval(
            logml_values, coverage=signed_error_coverage
        )
        posterior_counts[model_index] = len(posterior_values)
        logml_counts[model_index] = len(logml_values)
    # --------------------------------------------------
    # PMP
    # Combine benchmark datasets from M0-M3.
    # Each PMP component keeps its own interval.
    # --------------------------------------------------
    pmp_parts = []
    for source in ASSUMED_MODELS:
        metrics = metrics_by_source[source]
        valid = np.asarray(metrics["all_converged"], dtype=bool)
        values = np.asarray(metrics["signed_pmp_error"], dtype=np.float64)
        valid &= np.all(np.isfinite(values), axis=1)
        pmp_parts.append(values[valid])
    all_pmp_errors = np.concatenate(pmp_parts, axis=0)
    if len(all_pmp_errors) == 0:
        raise ValueError("No valid PMP errors for threshold calibration")
    (pmp_error_lower, pmp_error_upper) = central_interval(all_pmp_errors, coverage=signed_error_coverage, axis=0)
    return {
        "candidate_models": np.asarray(ASSUMED_MODELS),
        "posterior_mmd_lower": np.zeros(num_models, dtype=np.float64),
        "posterior_mmd_upper": (posterior_mmd_upper),
        "logml_error_lower": (logml_error_lower),
        "logml_error_upper": (logml_error_upper),
        "pmp_error_lower": (pmp_error_lower),
        "pmp_error_upper": (pmp_error_upper),
        "posterior_count": (posterior_counts),
        "logml_count": (logml_counts),
        "pmp_count": np.asarray(len(all_pmp_errors), dtype=np.int64),
    }


# ---------------------------------------------------------------------
# Direct thresholds
# ---------------------------------------------------------------------


def calibrate_direct_thresholds(metrics_by_source, *, signed_error_coverage=0.90):
    """
    Calibrate direct PMP-error thresholds.

    Benchmark datasets from M0-M3 are combined.

    Only datasets for which all four candidate
    Stan fits converged are retained.

    A separate central interval is calculated
    for each PMP component.
    """
    pmp_parts = []
    for source in ASSUMED_MODELS:
        metrics = metrics_by_source[source]
        valid = np.asarray(metrics["all_converged"], dtype=bool)
        values = np.asarray(metrics["signed_pmp_error"], dtype=np.float64)
        valid &= np.all(np.isfinite(values), axis=1)
        pmp_parts.append(values[valid])
    all_pmp_errors = np.concatenate(pmp_parts, axis=0)
    if len(all_pmp_errors) == 0:
        raise ValueError("No valid direct PMP errors for threshold calibration")
    (pmp_error_lower, pmp_error_upper) = central_interval(all_pmp_errors, coverage=signed_error_coverage, axis=0)
    return {
        "candidate_models": np.asarray(ASSUMED_MODELS),
        "pmp_error_lower": (pmp_error_lower),
        "pmp_error_upper": (pmp_error_upper),
        "pmp_count": np.asarray(len(all_pmp_errors), dtype=np.int64),
    }


# ---------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------


def run_indirect_thresholds(
    *, summary_multiplier, posterior_quantile=0.95, signed_error_coverage=0.90, overwrite=False
):
    """Calibrate and save indirect thresholds."""
    metrics_by_source = load_all_metrics(method="indirect", summary_multiplier=summary_multiplier)
    thresholds = calibrate_indirect_thresholds(
        metrics_by_source, posterior_quantile=posterior_quantile, signed_error_coverage=signed_error_coverage
    )
    output_path = threshold_path(method="indirect", summary_multiplier=summary_multiplier)
    save_thresholds(thresholds, output_path, overwrite=overwrite)
    print(f"Saved: {output_path}")
    return thresholds

def run_direct_thresholds(*, scoring_rule, signed_error_coverage=0.90, overwrite=False):
    """Calibrate and save direct PMP thresholds."""
    metrics_by_source = load_all_metrics(method="direct", scoring_rule=scoring_rule)
    thresholds = calibrate_direct_thresholds(metrics_by_source, signed_error_coverage=signed_error_coverage)
    output_path = threshold_path(method="direct", scoring_rule=scoring_rule)
    save_thresholds(thresholds, output_path, overwrite=overwrite)
    print(f"Saved: {output_path}")
    return thresholds

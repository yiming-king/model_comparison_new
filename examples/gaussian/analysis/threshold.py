"""Calibrate inference-error thresholds from benchmark datasets."""
# fmt: off

from pathlib import Path

import numpy as np

from ..config import ASSUMED_MODELS, RESULT_DIR
from .metrics import load_npz, metrics_path

THRESHOLD_DIR = RESULT_DIR / "thresholds"

def central_interval(values, *, coverage=0.90, axis=0):
    """Return the central empirical interval."""
    values = np.asarray(values, dtype=np.float64)
    alpha = 1.0 - coverage
    lower = np.quantile(values, alpha / 2, axis=axis)
    upper = np.quantile(values, 1.0 - alpha / 2, axis=axis)
    return lower, upper

def upper_quantile(values, *, quantile=0.95, axis=0):
    """Return an upper empirical quantile."""
    values = np.asarray(values, dtype=np.float64)
    return np.quantile(values, quantile, axis=axis)

def threshold_path(*, method, num_obs, summary_dim, scoring_rule=None):
    """Return path for one calibrated threshold file."""
    if method == "indirect":
        name = f"indirect_s{summary_dim}.npz"
    elif method == "direct":
        if scoring_rule is None:
            raise ValueError("scoring_rule is required for direct thresholds")
        name = f"direct_{scoring_rule}_s{summary_dim}.npz"
    else:
        raise ValueError("method must be 'indirect' or 'direct'")
    return THRESHOLD_DIR / f"n{num_obs}" / name

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

def load_all_metrics(*, method, split, num_obs, summary_dim, scoring_rule=None):
    """
    Load metric files for all four assumed generating models.

    Returns:
        {
            "m1": metrics,
            "m2": metrics,
            "m3": metrics,
            "m4": metrics,
        }
    """
    results = {}
    for source_model in ASSUMED_MODELS:
        path = metrics_path(
            method=method,
            split=split,
            source_model=source_model,
            num_obs=num_obs,
            summary_dim=summary_dim,
            scoring_rule=scoring_rule,
        )
        results[source_model] = load_npz(path)
    return results

def calibrate_indirect_thresholds(metrics_by_source, *, posterior_quantile=0.95, signed_error_coverage=0.90):
    """
    Calibrate indirect inference-error thresholds.

    Posterior MMD:
        matching source/candidate model only
        [0, q95]

    logML:
        matching source/candidate model only
        central 90%

    PMP:
        all assumed-source benchmark datasets
        central 90% for each PMP component
    """
    num_models = len(ASSUMED_MODELS)
    posterior_mmd_upper = np.empty(num_models, dtype=np.float64)
    logml_error_lower = np.empty(num_models, dtype=np.float64)
    logml_error_upper = np.empty(num_models, dtype=np.float64)
    # Posterior MMD and logML thresholds:
    # use matching model only.
    for model_index, model in enumerate(ASSUMED_MODELS):
        metrics = metrics_by_source[model]
        posterior_values = metrics["posterior_mmd"][:, model_index]
        logml_values = metrics["signed_logml_error"][:, model_index]
        posterior_mmd_upper[model_index] = upper_quantile(posterior_values, quantile=posterior_quantile)
        (logml_error_lower[model_index], logml_error_upper[model_index]) = central_interval(
            logml_values, coverage=signed_error_coverage
        )
    # PMP is a model-comparison quantity.
    # Combine benchmark datasets from M1-M4.
    all_pmp_errors = np.concatenate([metrics_by_source[model]["signed_pmp_error"] for model in ASSUMED_MODELS], axis=0)
    (pmp_error_lower, pmp_error_upper) = central_interval(all_pmp_errors, coverage=signed_error_coverage, axis=0)
    return {
        "candidate_models": np.asarray(ASSUMED_MODELS),
        "posterior_mmd_lower": np.zeros(num_models, dtype=np.float64),
        "posterior_mmd_upper": (posterior_mmd_upper),
        "logml_error_lower": (logml_error_lower),
        "logml_error_upper": (logml_error_upper),
        "pmp_error_lower": (pmp_error_lower),
        "pmp_error_upper": (pmp_error_upper),
    }

def calibrate_direct_thresholds(metrics_by_source, *, signed_error_coverage=0.90):
    """
    Calibrate direct PMP-error thresholds.

    All benchmark datasets from M1-M4 are combined,
    while thresholds remain separate for each PMP component.
    """
    all_pmp_errors = np.concatenate([metrics_by_source[model]["signed_pmp_error"] for model in ASSUMED_MODELS], axis=0)
    (pmp_error_lower, pmp_error_upper) = central_interval(all_pmp_errors, coverage=signed_error_coverage, axis=0)
    return {
        "candidate_models": np.asarray(ASSUMED_MODELS),
        "pmp_error_lower": (pmp_error_lower),
        "pmp_error_upper": (pmp_error_upper),
    }

def run_indirect_thresholds(
    *, num_obs, summary_dim, split="benchmark", posterior_quantile=0.95, signed_error_coverage=0.90, overwrite=False
):
    """Calibrate and save indirect thresholds."""
    metrics_by_source = load_all_metrics(method="indirect", split=split, num_obs=num_obs, summary_dim=summary_dim)
    thresholds = calibrate_indirect_thresholds(
        metrics_by_source, posterior_quantile=posterior_quantile, signed_error_coverage=signed_error_coverage
    )
    output_path = threshold_path(method="indirect", num_obs=num_obs, summary_dim=summary_dim)
    save_thresholds(thresholds, output_path, overwrite=overwrite)
    print(f"Saved: {output_path}")
    return thresholds

def run_direct_thresholds(
    *, num_obs, summary_dim, scoring_rule, split="benchmark", signed_error_coverage=0.90, overwrite=False
):
    """Calibrate and save direct PMP thresholds."""
    metrics_by_source = load_all_metrics(
        method="direct", split=split, num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule
    )
    thresholds = calibrate_direct_thresholds(metrics_by_source, signed_error_coverage=signed_error_coverage)
    output_path = threshold_path(method="direct", num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
    save_thresholds(thresholds, output_path, overwrite=overwrite)
    print(f"Saved: {output_path}")
    return thresholds

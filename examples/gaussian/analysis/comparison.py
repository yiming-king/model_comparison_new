"""Combine diagnostic scores with inference errors."""
# fmt: off

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..config import ASSUMED_MODELS, SOURCE_MODELS, RESULT_DIR, get_result_dir
from .diagnostic import DIAGNOSTICS, diagnostic_result_path, diagnostic_threshold_path
from .inference import inference_path
from .metric import gold_path, load_npz, metrics_path
from .threshold import threshold_path

COMPARISON_DIR = RESULT_DIR / "comparison"

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def comparison_root(*, method, num_obs, summary_dim, scoring_rule=None, variant="baseline"):
    if method == "indirect":
        name = f"indirect_s{summary_dim}"
        root = get_result_dir(variant) / "comparison"
    elif method == "direct":
        if scoring_rule is None:
            raise ValueError("scoring_rule is required for direct comparison")
        name = f"direct_{scoring_rule}_s{summary_dim}"
    else:
        raise ValueError("method must be 'indirect' or 'direct'")
    return (root if method == "indirect" else COMPARISON_DIR) / f"n{num_obs}" / name

def comparison_paths(*, method, num_obs, summary_dim, scoring_rule=None, variant="baseline"):
    root = comparison_root(method=method, num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule, variant=variant)
    return {
        "data": root / "data.csv",
        "detection": root / "detection.csv",
        "global_pmp_detection": (root / "global_pmp_detection.csv"),
    }

def save_csv(frame, path, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def outside_interval(values, lower, upper):
    values = np.asarray(values)
    return (values < lower) | (values > upper)

def safe_ratio(numerator, denominator):
    denominator = float(denominator)
    if np.isclose(denominator, 0.0):
        return np.nan
    return float(numerator) / denominator

def check_alignment(metrics, diagnostics, inference, gold):
    """Check that all files refer to the same datasets."""
    ids = metrics["id"]
    sources = metrics["source_model"]
    for name, values in (("diagnostics", diagnostics), ("inference", inference), ("gold", gold)):
        if not np.array_equal(ids, values["id"]):
            raise ValueError(f"Dataset IDs do not match: {name}")
        if not np.array_equal(sources, values["source_model"]):
            raise ValueError(f"Source models do not match: {name}")
    for name, values in (("inference", inference), ("gold", gold)):
        if not np.array_equal(metrics["candidate_models"], values["candidate_models"]):
            raise ValueError(f"Candidate models do not match: {name}")

# ---------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------

def binary_metrics(actual, predicted, score):
    """Compute TP, FP, FN, TN, FNR, FPR, and ROC AUC."""
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted))
    fnr = fn / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    auc = float(roc_auc_score(actual, score)) if np.unique(actual).size == 2 else np.nan
    return {
        "n": len(actual),
        "n_error_positive": int(actual.sum()),
        "n_error_negative": int((~actual).sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "FNR": fnr,
        "FPR": fpr,
        "AUC": auc,
    }

def component_detection(data, *, method, error_metrics):
    """
    Evaluate diagnostics against model-specific inference errors.

    Diagnostic positive:
        rho > 1

    This corresponds to high surprise, i.e. exceeding the
    calibrated upper diagnostic threshold.
    """
    error_specs = {
        "posterior_mmd": {
            "positive": "posterior_error_positive",
            "lower": "posterior_error_lower",
            "upper": "posterior_error_upper",
        },
        "logml": {"positive": "logml_error_positive", "lower": "logml_error_lower", "upper": "logml_error_upper"},
        "pmp": {"positive": "pmp_error_positive", "lower": "pmp_error_lower", "upper": "pmp_error_upper"},
    }
    rows = []
    for model in ASSUMED_MODELS:
        model_data = data.loc[data["candidate_model"].eq(model)]
        for error_metric in error_metrics:
            spec = error_specs[error_metric]
            actual = model_data[spec["positive"]].to_numpy(bool)
            for diagnostic in DIAGNOSTICS:
                rho = model_data[f"rho_{diagnostic}"].to_numpy(float)
                predicted = rho > 1.0
                result = binary_metrics(actual, predicted, rho)
                rows.append(
                    {
                        "method": method,
                        "scope": "component",
                        "candidate_model": model,
                        "error_metric": error_metric,
                        "diagnostic": diagnostic,
                        "error_lower": float(model_data[spec["lower"]].iloc[0]),
                        "error_upper": float(model_data[spec["upper"]].iloc[0]),
                        "diagnostic_lower": float(model_data[f"{diagnostic}_lower"].iloc[0]),
                        "diagnostic_upper": float(model_data[f"{diagnostic}_upper"].iloc[0]),
                        "rho_low": float(model_data[f"rho_low_{diagnostic}"].iloc[0]),
                        "rho_threshold": 1.0,
                        **result,
                    }
                )
    return pd.DataFrame(rows)

def global_pmp_detection(data, *, method):
    """
    Dataset-level PMP detection.

    Actual positive:
        at least one PMP component exceeds its error interval.

    Indirect:
        diagnostic score = max rho across M1-M4.

    Direct:
        one shared rho, so max leaves it unchanged.
    """
    rows = []
    for diagnostic in DIAGNOSTICS:
        grouped = (
            data.groupby(["source_model", "id"], sort=False)
            .agg(high_pmp_error=("high_pmp_error_any", "first"), rho=(f"rho_{diagnostic}", "max"))
            .reset_index()
        )
        actual = grouped["high_pmp_error"].to_numpy(bool)
        rho = grouped["rho"].to_numpy(float)
        predicted = rho > 1.0
        result = binary_metrics(actual, predicted, rho)
        rows.append(
            {
                "method": method,
                "scope": "global_pmp",
                "candidate_model": "any",
                "error_metric": "pmp",
                "diagnostic": diagnostic,
                "rho_threshold": 1.0,
                **result,
            }
        )
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Indirect
# ---------------------------------------------------------------------

def build_indirect_comparison(*, num_obs, summary_dim, split="simulated", sources=SOURCE_MODELS, variant="baseline"):
    """Build long-form indirect comparison data."""
    error_thresholds = load_npz(threshold_path(method="indirect", num_obs=num_obs, summary_dim=summary_dim, variant=variant))
    diagnostic_thresholds = load_npz(
        diagnostic_threshold_path(method="indirect", num_obs=num_obs, summary_dim=summary_dim, variant=variant)
    )
    frames = []
    for source in sources:
        metrics = load_npz(
            metrics_path(method="indirect", split=split, source_model=source, num_obs=num_obs, summary_dim=summary_dim,
                variant=variant)
        )
        diagnostics = load_npz(
            diagnostic_result_path(
                method="indirect", split=split, source_model=source, num_obs=num_obs, summary_dim=summary_dim,
                variant=variant
            )
        )
        inference = load_npz(
            inference_path(
                method="indirect", split=split, source_model=source, num_obs=num_obs, summary_dim=summary_dim,
                variant=variant
            )
        )
        gold = load_npz(gold_path(split=split, source_model=source, num_obs=num_obs))
        check_alignment(metrics, diagnostics, inference, gold)
        pmp_errors = metrics["signed_pmp_error"]
        high_pmp_error_any = outside_interval(
            pmp_errors, error_thresholds["pmp_error_lower"][None, :], error_thresholds["pmp_error_upper"][None, :]
        ).any(axis=1)
        for model_index, model in enumerate(ASSUMED_MODELS):
            posterior_error = metrics["posterior_mmd"][:, model_index]
            logml_error = metrics["signed_logml_error"][:, model_index]
            pmp_error = metrics["signed_pmp_error"][:, model_index]
            posterior_lower = float(error_thresholds["posterior_mmd_lower"][model_index])
            posterior_upper = float(error_thresholds["posterior_mmd_upper"][model_index])
            logml_lower = float(error_thresholds["logml_error_lower"][model_index])
            logml_upper = float(error_thresholds["logml_error_upper"][model_index])
            pmp_lower = float(error_thresholds["pmp_error_lower"][model_index])
            pmp_upper = float(error_thresholds["pmp_error_upper"][model_index])
            frame = pd.DataFrame(
                {
                    "method": "indirect",
                    "num_obs": num_obs,
                    "summary_dim": summary_dim,
                    "source_model": source,
                    "source_is_assumed": (source in ASSUMED_MODELS),
                    "id": metrics["id"],
                    "candidate_model": model,
                    # Posterior
                    "posterior_mmd": posterior_error,
                    "posterior_error_lower": posterior_lower,
                    "posterior_error_upper": posterior_upper,
                    "posterior_error_positive": (posterior_error > posterior_upper),
                    # logML
                    "estimated_logml": (inference["logml"][:, model_index]),
                    "gold_logml": (gold["logml"][:, model_index]),
                    "signed_logml_error": logml_error,
                    "absolute_logml_error": (metrics["absolute_logml_error"][:, model_index]),
                    "logml_error_lower": logml_lower,
                    "logml_error_upper": logml_upper,
                    "logml_error_positive": (outside_interval(logml_error, logml_lower, logml_upper)),
                    # PMP
                    "estimated_pmp": (inference["pmp"][:, model_index]),
                    "gold_pmp": (gold["pmp"][:, model_index]),
                    "signed_pmp_error": pmp_error,
                    "absolute_pmp_error": (metrics["absolute_pmp_error"][:, model_index]),
                    "pmp_error_lower": pmp_lower,
                    "pmp_error_upper": pmp_upper,
                    "pmp_error_positive": (outside_interval(pmp_error, pmp_lower, pmp_upper)),
                    "pmp_l1_error": (metrics["pmp_l1_error"]),
                    "high_pmp_error_any": (high_pmp_error_any),
                    # IS quality
                    "importance_ess": (metrics["importance_ess"][:, model_index]),
                    "importance_ess_ratio": (metrics["importance_ess_ratio"][:, model_index]),
                }
            )
            for diagnostic in DIAGNOSTICS:
                score = diagnostics[f"d_{diagnostic}"][:, model_index]
                rho = diagnostics[f"rho_{diagnostic}"][:, model_index]
                lower = float(diagnostic_thresholds[f"d_{diagnostic}_lower"][model_index])
                upper = float(diagnostic_thresholds[f"d_{diagnostic}_upper"][model_index])
                frame[f"d_{diagnostic}"] = score
                frame[f"rho_{diagnostic}"] = rho
                frame[f"{diagnostic}_lower"] = lower
                frame[f"{diagnostic}_upper"] = upper
                frame[f"rho_low_{diagnostic}"] = safe_ratio(lower, upper)
                # Central calibration region
                frame[f"typical_{diagnostic}"] = (score >= lower) & (score <= upper)
                # Detection rule used for FNR/FPR/AUC
                frame[f"high_surprise_{diagnostic}"] = rho > 1.0
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)

def run_indirect_comparison(*, num_obs, summary_dim, split="simulated", sources=SOURCE_MODELS, overwrite=False, variant="baseline"):
    data = build_indirect_comparison(num_obs=num_obs, summary_dim=summary_dim, split=split, sources=sources, variant=variant)
    detection = component_detection(data, method="indirect", error_metrics=("posterior_mmd", "logml", "pmp"))
    global_detection = global_pmp_detection(data, method="indirect")
    paths = comparison_paths(method="indirect", num_obs=num_obs, summary_dim=summary_dim, variant=variant)
    save_csv(data, paths["data"], overwrite=overwrite)
    save_csv(detection, paths["detection"], overwrite=overwrite)
    save_csv(global_detection, paths["global_pmp_detection"], overwrite=overwrite)
    print(f"Saved indirect comparison: {paths['data'].parent}")
    return {"data": data, "detection": detection, "global_pmp_detection": global_detection}

# ---------------------------------------------------------------------
# Direct
# ---------------------------------------------------------------------

def build_direct_comparison(*, num_obs, summary_dim, scoring_rule, split="simulated", sources=SOURCE_MODELS):
    """Build long-form direct comparison data."""
    error_thresholds = load_npz(
        threshold_path(method="direct", num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
    )
    diagnostic_thresholds = load_npz(
        diagnostic_threshold_path(method="direct", num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
    )
    frames = []
    for source in sources:
        metrics = load_npz(
            metrics_path(
                method="direct",
                split=split,
                source_model=source,
                num_obs=num_obs,
                summary_dim=summary_dim,
                scoring_rule=scoring_rule,
            )
        )
        diagnostics = load_npz(
            diagnostic_result_path(
                method="direct",
                split=split,
                source_model=source,
                num_obs=num_obs,
                summary_dim=summary_dim,
                scoring_rule=scoring_rule,
            )
        )
        inference = load_npz(
            inference_path(
                method="direct",
                split=split,
                source_model=source,
                num_obs=num_obs,
                summary_dim=summary_dim,
                scoring_rule=scoring_rule,
            )
        )
        gold = load_npz(gold_path(split=split, source_model=source, num_obs=num_obs))
        check_alignment(metrics, diagnostics, inference, gold)
        pmp_errors = metrics["signed_pmp_error"]
        high_pmp_error_any = outside_interval(
            pmp_errors, error_thresholds["pmp_error_lower"][None, :], error_thresholds["pmp_error_upper"][None, :]
        ).any(axis=1)
        for model_index, model in enumerate(ASSUMED_MODELS):
            pmp_error = pmp_errors[:, model_index]
            pmp_lower = float(error_thresholds["pmp_error_lower"][model_index])
            pmp_upper = float(error_thresholds["pmp_error_upper"][model_index])
            frame = pd.DataFrame(
                {
                    "method": "direct",
                    "scoring_rule": scoring_rule,
                    "num_obs": num_obs,
                    "summary_dim": summary_dim,
                    "source_model": source,
                    "source_is_assumed": (source in ASSUMED_MODELS),
                    "id": metrics["id"],
                    "candidate_model": model,
                    "estimated_pmp": (inference["pmp"][:, model_index]),
                    "gold_pmp": (gold["pmp"][:, model_index]),
                    "signed_pmp_error": pmp_error,
                    "absolute_pmp_error": (metrics["absolute_pmp_error"][:, model_index]),
                    "pmp_error_lower": pmp_lower,
                    "pmp_error_upper": pmp_upper,
                    "pmp_error_positive": (outside_interval(pmp_error, pmp_lower, pmp_upper)),
                    "pmp_l1_error": (metrics["pmp_l1_error"]),
                    "high_pmp_error_any": (high_pmp_error_any),
                }
            )
            # Direct uses one shared diagnostic space.
            # Therefore these values are identical for all
            # candidate-model rows belonging to one dataset.
            for diagnostic in DIAGNOSTICS:
                score = diagnostics[f"d_{diagnostic}"]
                rho = diagnostics[f"rho_{diagnostic}"]
                lower = float(diagnostic_thresholds[f"d_{diagnostic}_lower"])
                upper = float(diagnostic_thresholds[f"d_{diagnostic}_upper"])
                frame[f"d_{diagnostic}"] = score
                frame[f"rho_{diagnostic}"] = rho
                frame[f"{diagnostic}_lower"] = lower
                frame[f"{diagnostic}_upper"] = upper
                frame[f"rho_low_{diagnostic}"] = safe_ratio(lower, upper)
                frame[f"typical_{diagnostic}"] = (score >= lower) & (score <= upper)
                frame[f"high_surprise_{diagnostic}"] = rho > 1.0
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)

def run_direct_comparison(
    *, num_obs, summary_dim, scoring_rule, split="simulated", sources=SOURCE_MODELS, overwrite=False
):
    data = build_direct_comparison(
        num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule, split=split, sources=sources
    )
    detection = component_detection(data, method="direct", error_metrics=("pmp",))
    global_detection = global_pmp_detection(data, method="direct")
    paths = comparison_paths(method="direct", num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
    save_csv(data, paths["data"], overwrite=overwrite)
    save_csv(detection, paths["detection"], overwrite=overwrite)
    save_csv(global_detection, paths["global_pmp_detection"], overwrite=overwrite)
    print(f"Saved direct comparison: {paths['data'].parent}")
    return {"data": data, "detection": detection, "global_pmp_detection": global_detection}

# ---------------------------------------------------------------------
# Load saved results
# ---------------------------------------------------------------------

def load_comparison(*, method, num_obs, summary_dim, scoring_rule=None, variant="baseline"):
    paths = comparison_paths(method=method, num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule, variant=variant)
    return {name: pd.read_csv(path) for name, path in paths.items()}

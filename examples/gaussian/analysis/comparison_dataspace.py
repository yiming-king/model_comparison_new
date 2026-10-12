"""Append data-space diagnostics to EXISTING summary-space comparison CSVs.

Never runs NPE inference, computes errors, or changes original CSVs.
The output preserves all original columns and adds only data_* columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..config import ASSUMED_MODELS, SOURCE_MODELS, RESULT_DIR
from .diagnostic_dataspace import (
    DIAGNOSTICS, diagnostic_result_path, diagnostic_threshold_path,
    load_npz, load_vectors, validate_metadata, validate_thresholds,
    file_digest,
)

KEYS = ["source_model", "id", "candidate_model"]
ERROR_FLAGS = {
    "posterior_mmd": "posterior_error_positive",
    "logml": "logml_error_positive",
    "pmp": "pmp_error_positive",
}


def result_root(variant="baseline"):
    if not variant or variant in (".", "..") or "/" in variant or "\\" in variant:
        raise ValueError(f"Invalid variant: {variant!r}")
    return RESULT_DIR if variant == "baseline" else RESULT_DIR / "ablation" / variant


def output_root(variant="baseline"):
    if not variant or variant in (".", "..") or "/" in variant or "\\" in variant:
        raise ValueError(f"Invalid variant: {variant!r}")
    root = RESULT_DIR / "data_space"
    return root if variant == "baseline" else root / "ablation" / variant


def comparison_path(*, num_obs, summary_dim, variant="baseline"):
    run = f"comparison/n{num_obs}/indirect_s{summary_dim}/data.csv"
    return result_root(variant) / run, output_root(variant) / run


def load_data_diagnostics(num_obs):
    """Return (source_model,id,candidate_model)-keyed data-space columns.

    No network variant or summary_dim enters here.
    """
    thresholds = validate_thresholds(num_obs=num_obs)
    signature = file_digest(diagnostic_threshold_path(num_obs=num_obs))
    frames = []
    for source in SOURCE_MODELS:
        raw = load_vectors(num_obs=num_obs, split="simulated", source_model=source)
        path = diagnostic_result_path(num_obs=num_obs, split="simulated", source_model=source)
        saved = load_npz(path)
        validate_metadata(saved, num_obs=num_obs)
        if (str(saved["dataset_digest"]) != raw["dataset_digest"]
                or str(saved["thresholds_digest"]) != signature):
            raise ValueError(f"Stale data-space diagnostics: {path}")
        if (not np.array_equal(saved["id"], raw["id"])
                or not np.array_equal(saved["source_model"], raw["source_model"])):
            raise ValueError(f"Dataset identities differ: {path}")

        for index, candidate in enumerate(ASSUMED_MODELS):
            frame = pd.DataFrame({
                "source_model": source, "id": saved["id"], "candidate_model": candidate,
                "data_diagnostic_dim": int(saved["diagnostic_dim"]),
            })
            for metric in DIAGNOSTICS:
                matrix = saved[f"d_{metric}"]
                if matrix.shape != (len(raw["id"]), len(ASSUMED_MODELS)):
                    raise ValueError(f"Wrong score shape in {path}: {metric}")
                low = float(thresholds[f"d_{metric}_lower"][index])
                high = float(thresholds[f"d_{metric}_upper"][index])
                score = np.asarray(matrix[:, index], float)
                if not (np.isfinite(score).all() and np.isfinite([low, high]).all() and high > 0):
                    raise ValueError(f"Invalid diagnostic scores/thresholds in {path}: {metric}")
                rho = score / high
                frame[f"data_d_{metric}"] = score
                frame[f"data_{metric}_lower"] = low
                frame[f"data_{metric}_upper"] = high
                frame[f"data_rho_{metric}"] = rho
                frame[f"data_rho_low_{metric}"] = low / high
                frame[f"data_typical_{metric}"] = (score >= low) & (score <= high)
                frame[f"data_high_surprise_{metric}"] = rho > 1.0
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _binary_counts(actual, score):
    actual, score = np.asarray(actual, bool), np.asarray(score, float)
    predicted = score > 1.0
    tp = int((actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    fp = int((~actual & predicted).sum())
    tn = int((~actual & ~predicted).sum())
    return {
        "n": len(actual), "n_error_positive": int(actual.sum()),
        "n_error_negative": int((~actual).sum()),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "FNR": fn / (tp + fn) if tp + fn else np.nan,
        "FPR": fp / (fp + tn) if fp + tn else np.nan,
        "AUC": roc_auc_score(actual, score) if np.unique(actual).size == 2 else np.nan,
    }


def make_detection_tables(data):
    """Keep original error flags; only replace the predictor with data-space rho."""
    components = []
    for model in ASSUMED_MODELS:
        rows = data.loc[data.candidate_model.eq(model)]
        for error_name, positive in ERROR_FLAGS.items():
            for metric in DIAGNOSTICS:
                components.append({
                    "method": "indirect", "diagnostic_space": "data",
                    "candidate_model": model, "error_metric": error_name, "diagnostic": metric,
                    **_binary_counts(rows[positive].to_numpy(bool), rows[f"data_rho_{metric}"]),
                })
    global_rows = []
    for metric in DIAGNOSTICS:
        grouped = data.groupby(["source_model", "id"], sort=False).agg(
            actual=("high_pmp_error_any", "first"), rho=(f"data_rho_{metric}", "max")
        )
        global_rows.append({
            "method": "indirect", "diagnostic_space": "data",
            "candidate_model": "any", "error_metric": "pmp", "diagnostic": metric,
            **_binary_counts(grouped.actual, grouped.rho),
        })
    return pd.DataFrame(components), pd.DataFrame(global_rows)


def run_comparison(*, num_obs, summary_dim, variant="baseline"):
    """Append new columns to the original CSV, writing to data_space/ only."""
    source, target = comparison_path(num_obs=num_obs, summary_dim=summary_dim, variant=variant)
    original = pd.read_csv(source)
    if not (original.method.eq("indirect").all()
            and original.num_obs.eq(num_obs).all()
            and original.summary_dim.eq(summary_dim).all()):
        raise ValueError(f"Incorrect network or dataset metadata in {source}")
    if original.empty or original.duplicated(KEYS).any():
        raise ValueError(f"Empty or duplicate dataset/model rows in {source}")
    if any(column.startswith("data_") for column in original.columns):
        raise ValueError(f"Source CSV already has data-space columns: {source}")

    added = load_data_diagnostics(num_obs)
    if added.duplicated(KEYS).any():
        raise ValueError("Duplicate diagnostic row keys")
    # Check full key coverage, not just the total number of rows.
    old_keys = pd.MultiIndex.from_frame(original[KEYS])
    new_keys = pd.MultiIndex.from_frame(added[KEYS])
    if len(old_keys.difference(new_keys)) or len(new_keys.difference(old_keys)):
        raise ValueError("Old CSV and new diagnostics refer to different dataset/model keys")

    combined = original.merge(added, on=KEYS, how="left", validate="one_to_one", sort=False)
    # An explicit invariant: no legacy metric or summary-space diagnostic changes.
    pd.testing.assert_frame_equal(original, combined[original.columns], check_exact=True)
    for metric in DIAGNOSTICS:
        combined[f"data_all_high_surprise_{metric}"] = (
            combined.groupby(["source_model", "id"])[f"data_high_surprise_{metric}"].transform("all")
        )
        combined[f"data_at_least_one_not_high_surprise_{metric}"] = (
            ~combined[f"data_all_high_surprise_{metric}"]
        )

    detection, global_detection = make_detection_tables(combined)
    target.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target, index=False)
    detection.to_csv(target.with_name("detection_dataspace.csv"), index=False)
    global_detection.to_csv(target.with_name("global_pmp_detection_dataspace.csv"), index=False)
    print(f"Combined CSV: {target}")
    return combined

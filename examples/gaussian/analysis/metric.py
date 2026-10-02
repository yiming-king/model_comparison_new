"""Compute inference errors against the analytical gold standard."""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from pathlib import Path

import numpy as np
import tensorflow as tf
from bayesflow.metrics import MaximumMeanDiscrepancy

from ..analytic.gold import GOLD_DIR
from ..config import RESULT_DIR, get_result_dir
from .inference import INFERENCE_DIR, inference_path

METRICS_DIR = RESULT_DIR / "metrics"

def load_npz(path):
    """Load an npz file as a dictionary."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def save_metrics(results, path, *, overwrite=False):
    """Save metric results."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Metric result already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **results)

def metrics_path(*, method, split, source_model, num_obs, summary_dim, scoring_rule=None, variant="baseline"):
    """Return metric path matching the corresponding inference path."""
    inf_path = inference_path(
        method=method,
        split=split,
        source_model=source_model,
        num_obs=num_obs,
        summary_dim=summary_dim,
        scoring_rule=scoring_rule,
        variant=variant,
    )
    root = get_result_dir(variant) if method == "indirect" else RESULT_DIR
    relative_path = inf_path.relative_to(root / "inference")
    return root / "metrics" / relative_path

def gold_path(*, split, source_model, num_obs):
    """Return the analytical gold-standard path."""
    return GOLD_DIR / f"n{num_obs}" / split / f"{source_model}.npz"

def check_alignment(inference, gold):
    """Make sure inference and gold refer to exactly the same datasets."""
    if not np.array_equal(inference["id"], gold["id"]):
        raise ValueError("Inference and gold dataset IDs do not match")
    if not np.array_equal(inference["source_model"], gold["source_model"]):
        raise ValueError("Inference and gold source models do not match")
    if not np.array_equal(inference["candidate_models"], gold["candidate_models"]):
        raise ValueError("Inference and gold candidate models do not match")

def compute_posterior_mmd(estimated_samples, gold_samples, *, num_samples=1024):
    """
    Compute posterior MMD for every dataset and candidate model.

    Input shapes:
        estimated_samples: (B, M, S, D)
        gold_samples:      (B, M, S, D)

    Output:
        (B, M)
    """
    estimated_samples = np.asarray(estimated_samples, dtype=np.float32)
    gold_samples = np.asarray(gold_samples, dtype=np.float32)
    num_datasets = estimated_samples.shape[0]
    num_models = estimated_samples.shape[1]
    sample_count = min(num_samples, estimated_samples.shape[2], gold_samples.shape[2])
    posterior_mmd = np.empty((num_datasets, num_models), dtype=np.float64)
    mmd = MaximumMeanDiscrepancy(kernel="gaussian")
    for dataset_index in range(num_datasets):
        for model_index in range(num_models):
            estimated = estimated_samples[dataset_index, model_index, :sample_count]
            gold = gold_samples[dataset_index, model_index, :sample_count]
            mmd2 = float(mmd(tf.convert_to_tensor(estimated), tf.convert_to_tensor(gold),))
            posterior_mmd[dataset_index, model_index] = np.sqrt(max(mmd2, 0.0))
    return posterior_mmd

def compute_indirect_metrics(inference, gold, *, num_mmd_samples=1024):
    """Compute posterior, logML, and PMP errors for indirect inference."""
    check_alignment(inference, gold)
    posterior_mmd = compute_posterior_mmd(
        inference["posterior_samples"], gold["posterior_samples"], num_samples=num_mmd_samples
    )
    signed_logml_error = inference["logml"] - gold["logml"]
    signed_pmp_error = inference["pmp"] - gold["pmp"]
    absolute_logml_error = np.abs(signed_logml_error)
    absolute_pmp_error = np.abs(signed_pmp_error)
    pmp_l1_error = np.sum(absolute_pmp_error, axis=1)
    num_importance_samples = inference["posterior_samples"].shape[2]
    importance_ess_ratio = inference["importance_ess"] / num_importance_samples
    return {
        "id": inference["id"],
        "source_model": inference["source_model"],
        "candidate_models": inference["candidate_models"],
        "posterior_mmd": posterior_mmd,
        "signed_logml_error": signed_logml_error,
        "absolute_logml_error": absolute_logml_error,
        "signed_pmp_error": signed_pmp_error,
        "absolute_pmp_error": absolute_pmp_error,
        "pmp_l1_error": pmp_l1_error,
        "importance_ess": inference["importance_ess"],
        "importance_ess_ratio": importance_ess_ratio,
    }

def compute_direct_metrics(inference, gold):
    """Compute PMP errors for direct model comparison."""
    check_alignment(inference, gold)
    signed_pmp_error = inference["pmp"] - gold["pmp"]
    absolute_pmp_error = np.abs(signed_pmp_error)
    pmp_l1_error = np.sum(absolute_pmp_error, axis=1)
    return {
        "id": inference["id"],
        "source_model": inference["source_model"],
        "candidate_models": inference["candidate_models"],
        "signed_pmp_error": signed_pmp_error,
        "absolute_pmp_error": absolute_pmp_error,
        "pmp_l1_error": pmp_l1_error,
    }

def run_indirect_metrics_file(*, split, source_model, num_obs, summary_dim, num_mmd_samples=1024, overwrite=False, variant="baseline"):
    """Compute and save indirect metrics for one inference file."""
    inf_path = inference_path(
        method="indirect", split=split, source_model=source_model, num_obs=num_obs, summary_dim=summary_dim,
        variant=variant
    )
    reference_path = gold_path(split=split, source_model=source_model, num_obs=num_obs)
    output_path = metrics_path(
        method="indirect", split=split, source_model=source_model, num_obs=num_obs, summary_dim=summary_dim,
        variant=variant
    )
    inference = load_npz(inf_path)
    gold = load_npz(reference_path)
    results = compute_indirect_metrics(inference, gold, num_mmd_samples=num_mmd_samples)
    save_metrics(results, output_path, overwrite=overwrite)
    print(f"Saved: {output_path}")

def run_direct_metrics_file(*, split, source_model, num_obs, summary_dim, scoring_rule, overwrite=False):
    """Compute and save direct PMP metrics for one inference file."""
    inf_path = inference_path(
        method="direct",
        split=split,
        source_model=source_model,
        num_obs=num_obs,
        summary_dim=summary_dim,
        scoring_rule=scoring_rule,
    )
    reference_path = gold_path(split=split, source_model=source_model, num_obs=num_obs)
    output_path = metrics_path(
        method="direct",
        split=split,
        source_model=source_model,
        num_obs=num_obs,
        summary_dim=summary_dim,
        scoring_rule=scoring_rule,
    )
    inference = load_npz(inf_path)
    gold = load_npz(reference_path)
    results = compute_direct_metrics(inference, gold)
    save_metrics(results, output_path, overwrite=overwrite)
    print(f"Saved: {output_path}")

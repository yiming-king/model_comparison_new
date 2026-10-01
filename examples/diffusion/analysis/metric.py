"""Compute diffusion inference errors against the Stan gold standard."""

from pathlib import Path

import numpy as np

from ..config import ASSUMED_MODELS, NUM_TRIALS, RESULT_DIR
from .gold import load_gold, load_posterior_draws
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

def metrics_path(*, method, split, source, summary_multiplier=None, scoring_rule=None):
    """Return metric path matching one inference file."""
    inf_path = inference_path(
        method=method, split=split, source=source, summary_multiplier=summary_multiplier, scoring_rule=scoring_rule
    )
    relative = inf_path.relative_to(INFERENCE_DIR)
    return METRICS_DIR / relative

def check_alignment(inference, gold):
    """Check that inference and gold use the same datasets."""
    if not np.array_equal(inference["id"], gold["id"]):
        raise ValueError("Inference and gold dataset IDs do not match")
    if not np.array_equal(inference["candidate_models"], gold["candidate_models"]):
        raise ValueError("Inference and gold candidate models do not match")
    if not np.array_equal(inference["generating_model"], gold["generating_model"]):
        raise ValueError("Inference and gold generating models do not match")


# ---------------------------------------------------------------------
# Posterior MMD
# ---------------------------------------------------------------------


def squared_distances(x, y):
    """Pairwise squared Euclidean distances."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    distances = np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2.0 * (x @ y.T)
    return np.maximum(distances, 0.0)

def rbf_bandwidth2(samples):
    """
    Median positive squared distance.

    The bandwidth is determined from the full Stan posterior.
    """
    distances = squared_distances(samples, samples)
    np.fill_diagonal(distances, 0.0)
    positive = distances[distances > 0.0]
    if len(positive) == 0:
        return 1.0
    return max(float(np.median(positive)), 1e-8)

def rbf_kernel(x, y, bandwidth2):
    """RBF kernel matrix."""
    bandwidth2 = max(float(bandwidth2), 1e-8)
    return np.exp(-squared_distances(x, y) / (2.0 * bandwidth2))

def squared_mmd_rbf(x, y, bandwidth2):
    """Biased squared RBF-MMD."""
    k_xx = rbf_kernel(x, x, bandwidth2)
    k_yy = rbf_kernel(y, y, bandwidth2)
    k_xy = rbf_kernel(x, y, bandwidth2)
    value = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return max(float(value), 0.0)

def compute_posterior_mmd(inference, *, split, source, num_samples=1024, seed=2025):
    """
    Compute posterior MMD for every dataset and candidate model.

    MMD is computed in the unconstrained parameter space:
        log(alpha), log(nu), logit(tau)
    """
    ids = inference["id"]
    num_datasets = len(ids)
    num_models = len(ASSUMED_MODELS)
    posterior_mmd = np.empty((num_datasets, num_models), dtype=np.float64)
    bandwidth2_values = np.empty((num_datasets, num_models), dtype=np.float64)
    for model_index, model in enumerate(ASSUMED_MODELS):
        npe_draws = np.asarray(inference[f"posterior_{model}"], dtype=np.float64)
        for dataset_index, dataset_id in enumerate(ids):
            stan_draws = load_posterior_draws(split, source, model, str(dataset_id))
            npe = npe_draws[dataset_index]
            if len(npe) < num_samples:
                raise ValueError(f"Need at least {num_samples} NPE draws for {source}/{dataset_id}/{model}")
            if len(stan_draws) < num_samples:
                raise ValueError(f"Need at least {num_samples} Stan draws for {source}/{dataset_id}/{model}")
            bandwidth2 = rbf_bandwidth2(stan_draws)
            rng = np.random.default_rng(seed + 100_000 + 1_000 * model_index + dataset_index)
            npe_index = rng.choice(len(npe), size=num_samples, replace=False)
            stan_index = rng.choice(len(stan_draws), size=num_samples, replace=False)
            npe_eval = npe[npe_index]
            stan_eval = stan_draws[stan_index]
            mmd2 = squared_mmd_rbf(npe_eval, stan_eval, bandwidth2)
            posterior_mmd[dataset_index, model_index] = np.sqrt(mmd2)
            bandwidth2_values[dataset_index, model_index] = bandwidth2
    return (posterior_mmd, bandwidth2_values)


# ---------------------------------------------------------------------
# Indirect metrics
# ---------------------------------------------------------------------


def compute_indirect_metrics(inference, gold, *, split, source, num_mmd_samples=1024, seed=2025):
    """Compute posterior, logML, and PMP errors."""
    check_alignment(inference, gold)
    (posterior_mmd, posterior_bandwidth2) = compute_posterior_mmd(
        inference, split=split, source=source, num_samples=num_mmd_samples, seed=seed
    )
    signed_logml_error = inference["logml"] - gold["logml"]
    signed_pmp_error = inference["pmp"] - gold["pmp"]
    absolute_logml_error = np.abs(signed_logml_error)
    absolute_pmp_error = np.abs(signed_pmp_error)
    pmp_l1_error = np.sum(absolute_pmp_error, axis=1)
    num_importance_samples = inference["posterior_m0"].shape[1]
    importance_ess_ratio = inference["importance_ess"] / num_importance_samples
    return {
        "id": inference["id"],
        "candidate_models": inference["candidate_models"],
        "generating_model": inference["generating_model"],
        "well_specified": inference["well_specified"],
        "posterior_mmd": posterior_mmd,
        "posterior_rbf_bandwidth2": (posterior_bandwidth2),
        "signed_logml_error": (signed_logml_error),
        "absolute_logml_error": (absolute_logml_error),
        "signed_pmp_error": (signed_pmp_error),
        "absolute_pmp_error": (absolute_pmp_error),
        "pmp_l1_error": pmp_l1_error,
        "importance_ess": inference["importance_ess"],
        "importance_ess_ratio": (importance_ess_ratio),
        "converged": gold["converged"],
        "all_converged": gold["all_converged"],
    }


# ---------------------------------------------------------------------
# Direct metrics
# ---------------------------------------------------------------------


def compute_direct_metrics(inference, gold):
    """Compute PMP error for direct model comparison."""
    check_alignment(inference, gold)
    signed_pmp_error = inference["pmp"] - gold["pmp"]
    absolute_pmp_error = np.abs(signed_pmp_error)
    pmp_l1_error = np.sum(absolute_pmp_error, axis=1)
    return {
        "id": inference["id"],
        "candidate_models": inference["candidate_models"],
        "generating_model": inference["generating_model"],
        "well_specified": inference["well_specified"],
        "signed_pmp_error": (signed_pmp_error),
        "absolute_pmp_error": (absolute_pmp_error),
        "pmp_l1_error": pmp_l1_error,
        "converged": gold["converged"],
        "all_converged": gold["all_converged"],
    }


# ---------------------------------------------------------------------
# File runners
# ---------------------------------------------------------------------


def run_indirect_metrics(*, split, source, summary_multiplier, num_mmd_samples=1024, seed=2025, overwrite=False):
    """Compute and save indirect metrics."""
    inf_path = inference_path(method="indirect", split=split, source=source, summary_multiplier=summary_multiplier)
    output_path = metrics_path(method="indirect", split=split, source=source, summary_multiplier=summary_multiplier)
    inference = load_npz(inf_path)
    gold = load_gold(split, source)
    results = compute_indirect_metrics(
        inference, gold, split=split, source=source, num_mmd_samples=num_mmd_samples, seed=seed
    )
    save_metrics(results, output_path, overwrite=overwrite)
    print(f"Saved: {output_path}")

def run_direct_metrics(*, split, source, scoring_rule, overwrite=False):
    """Compute and save direct PMP metrics."""
    inf_path = inference_path(method="direct", split=split, source=source, scoring_rule=scoring_rule)
    output_path = metrics_path(method="direct", split=split, source=source, scoring_rule=scoring_rule)
    inference = load_npz(inf_path)
    gold = load_gold(split, source)
    results = compute_direct_metrics(inference, gold)
    save_metrics(results, output_path, overwrite=overwrite)
    print(f"Saved: {output_path}")

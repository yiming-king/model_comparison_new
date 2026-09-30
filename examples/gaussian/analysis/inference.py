"""Run inference with trained Gaussian approximators."""
# fmt: off

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from pathlib import Path

import keras
import numpy as np

from ..analytic.gold import compute_pmp
from ..approximators.config import TrainingConfig, checkpoint_path
from ..approximators.estimation import MarginalLikelihoodEstimator
from ..config import ASSUMED_MODELS, DATASET_DIR, MODEL_SPECS, RESULT_DIR
from ..datasets.datasets import load_datasets

INFERENCE_DIR = RESULT_DIR / "inference"

def load_approximator(path):
    """Load a trained BayesFlow approximator."""
    return keras.saving.load_model(path)

def save_inference(results, path, *, overwrite=False):
    """Save inference results as a compressed npz file."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Inference result already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **results)

def inference_path(*, method, split, source_model, num_obs, summary_dim, scoring_rule=None):
    """Return output path for one inference result."""
    if method == "indirect":
        name = f"indirect_s{summary_dim}"
    elif method == "direct":
        if scoring_rule is None:
            raise ValueError("scoring_rule is required for direct inference")
        name = f"direct_{scoring_rule}_s{summary_dim}"
    else:
        raise ValueError("method must be 'indirect' or 'direct'")
    return INFERENCE_DIR / f"n{num_obs}" / split / name / f"{source_model}.npz"

def evaluate_indirect(datasets, *, num_obs, summary_dim, num_samples=2048, seed=2025):
    """Run indirect NPE inference for all assumed models."""
    x = np.asarray(datasets["x"], dtype=np.float32)
    ids = datasets["id"]
    source_model = datasets["source_model"]
    num_datasets, _, num_dims = x.shape
    num_models = len(ASSUMED_MODELS)
    posterior_samples = np.empty((num_datasets, num_models, num_samples, num_dims), dtype=np.float32)
    summaries = np.empty((num_datasets, num_models, summary_dim), dtype=np.float32)
    logml = np.empty((num_datasets, num_models), dtype=np.float64)
    importance_ess = np.empty((num_datasets, num_models), dtype=np.float64)
    for model_index, model in enumerate(ASSUMED_MODELS):
        print(f"Indirect inference: {model}, N={num_obs}, S={summary_dim}")
        config = TrainingConfig(
            num_dims=num_dims, num_obs=num_obs, summary_dim=summary_dim, summary_base_distribution="normal"
        )
        approximator = load_approximator(checkpoint_path(config, model=model))
        # Summary-space representation
        summaries[:, model_index] = approximator.summarize(conditions={"x": x})
        # Posterior samples
        samples = approximator.sample(conditions={"x": x}, num_samples=num_samples, seed=seed + model_index)
        model_samples = np.asarray(samples["mu"], dtype=np.float32)
        posterior_samples[:, model_index] = model_samples
        spec = MODEL_SPECS[model]
        # Marginal likelihood via importance sampling
        for dataset_index in range(num_datasets):
            estimator = MarginalLikelihoodEstimator(
                approximator=approximator,
                mu=model_samples[dataset_index],
                obs_data=x[dataset_index],
                mu_prior_mean=spec["mu_prior_mean"],
                mu_prior_std=spec["mu_prior_std"],
                num_dims=num_dims,
                likelihood_std=spec["likelihood_std"],
            )
            logml[dataset_index, model_index] = estimator.log_marginal_npe()
            importance_ess[dataset_index, model_index] = estimator.importance_ess
    pmp = compute_pmp(logml)
    return {
        "id": ids,
        "source_model": source_model,
        "candidate_models": np.asarray(ASSUMED_MODELS),
        "posterior_samples": posterior_samples,
        "summaries": summaries,
        "logml": logml,
        "pmp": pmp,
        "importance_ess": importance_ess,
    }

def evaluate_direct(datasets, *, num_obs, summary_dim, scoring_rule):
    """Run direct model-comparison inference."""
    x = np.asarray(datasets["x"], dtype=np.float32)
    config = TrainingConfig(num_dims=x.shape[-1], num_obs=num_obs, summary_dim=summary_dim)
    approximator = load_approximator(checkpoint_path(config, scoring_rule=scoring_rule))
    print(f"Direct inference: {scoring_rule}, N={num_obs}, S={summary_dim}")
    estimates = approximator.estimate(conditions={"x": x}, return_summaries=True)
    return {
        "id": datasets["id"],
        "source_model": datasets["source_model"],
        "candidate_models": np.asarray(ASSUMED_MODELS),
        "pmp": np.asarray(estimates["model_probs"]),
        "log_odds": np.asarray(estimates["log_odds"]),
        "summaries": np.asarray(estimates["_summaries"]),
    }

def run_indirect_file(*, split, source_model, num_obs, summary_dim, num_samples=2048, seed=2025, overwrite=False):
    """Run and save indirect inference for one stored dataset file."""
    dataset_path = DATASET_DIR / f"n{num_obs}" / split / f"{source_model}.npz"
    output_path = inference_path(
        method="indirect", split=split, source_model=source_model, num_obs=num_obs, summary_dim=summary_dim
    )
    datasets = load_datasets(dataset_path)
    results = evaluate_indirect(datasets, num_obs=num_obs, summary_dim=summary_dim, num_samples=num_samples, seed=seed)
    save_inference(results, output_path, overwrite=overwrite)
    print(f"Saved: {output_path}")

def run_direct_file(*, split, source_model, num_obs, summary_dim, scoring_rule, overwrite=False):
    """Run and save direct inference for one stored dataset file."""
    dataset_path = DATASET_DIR / f"n{num_obs}" / split / f"{source_model}.npz"
    output_path = inference_path(
        method="direct",
        split=split,
        source_model=source_model,
        num_obs=num_obs,
        summary_dim=summary_dim,
        scoring_rule=scoring_rule,
    )
    datasets = load_datasets(dataset_path)
    results = evaluate_direct(datasets, num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
    save_inference(results, output_path, overwrite=overwrite)
    print(f"Saved: {output_path}")

"""Run inference with trained diffusion approximators."""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from pathlib import Path

import keras
import numpy as np
from scipy.special import softmax

from ..approximators.config import TrainingConfig, checkpoint_path
from ..approximators.estimation import MarginalLikelihoodEstimator, parameter_array
from ..config import ASSUMED_MODELS, NUM_TRIALS, PARAM_DIMS, RESULT_DIR
from ..datasets import load_benchmark_dataset, load_simulated_dataset

INFERENCE_DIR = RESULT_DIR / "inference"

DIRECT_SUMMARY_DIM = 12

def data_conditions(data):
    data = np.asarray(data, dtype=np.float32)
    return {"rt": data[..., 0], "conditions": data[..., 1]}

def load_approximator(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trained approximator not found: {path}")
    return keras.saving.load_model(path)

def save_inference(results, path, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Inference result already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **results)

def inference_path(*, method, split, source, summary_multiplier=None, scoring_rule=None):
    if method == "indirect":
        if summary_multiplier is None:
            raise ValueError("summary_multiplier is required for indirect inference")
        name = f"indirect_s{summary_multiplier}D"
    elif method == "direct":
        if scoring_rule is None:
            raise ValueError("scoring_rule is required for direct inference")
        name = f"direct_{scoring_rule}_s{DIRECT_SUMMARY_DIM}"
    else:
        raise ValueError("method must be 'indirect' or 'direct'")
    return INFERENCE_DIR / f"n{NUM_TRIALS}" / split / name / f"{source}.npz"

def evaluate_indirect(datasets, *, summary_multiplier, num_samples=2048, seed=2025, batch_size=None):
    data = np.asarray(datasets["data"], dtype=np.float32)
    ids = np.asarray(datasets["ids"])
    num_datasets = len(data)
    num_models = len(ASSUMED_MODELS)
    logml = np.empty((num_datasets, num_models), dtype=np.float64)
    importance_ess = np.empty((num_datasets, num_models), dtype=np.float64)
    output = {
        "id": ids,
        "candidate_models": np.asarray(ASSUMED_MODELS),
        "generating_model": np.full(num_datasets, datasets["generating_model"]),
        "well_specified": np.full(num_datasets, datasets["well_specified"], dtype=bool),
    }
    conditions = data_conditions(data)
    for model_index, model in enumerate(ASSUMED_MODELS):
        summary_dim = PARAM_DIMS[model] * summary_multiplier
        print(f"Indirect inference: {model}, S={summary_dim}")
        config = TrainingConfig(summary_dim=summary_dim)
        approximator = load_approximator(checkpoint_path(config, model=model))
        samples = approximator.sample(
            conditions=conditions,
            num_samples=num_samples,
            sample_shape=(),
            return_summaries=True,
            batch_size=batch_size,
            seed=seed + model_index,
        )
        theta = parameter_array(samples)
        output[f"posterior_{model}"] = theta.astype(np.float32)
        output[f"summary_{model}"] = np.asarray(samples["_summaries"], dtype=np.float32)
        for dataset_index in range(num_datasets):
            estimator = MarginalLikelihoodEstimator(
                approximator=approximator,
                theta=theta[dataset_index],
                data=data[dataset_index],
                model=model,
                batch_size=batch_size,
            )
            logml[dataset_index, model_index] = estimator.log_marginal_npe()
            importance_ess[dataset_index, model_index] = estimator.importance_ess
    output["logml"] = logml
    output["pmp"] = softmax(logml, axis=-1)
    output["importance_ess"] = importance_ess
    return output

def evaluate_direct(datasets, *, scoring_rule):
    data = np.asarray(datasets["data"], dtype=np.float32)
    num_datasets = len(data)
    config = TrainingConfig(summary_dim=DIRECT_SUMMARY_DIM)
    approximator = load_approximator(checkpoint_path(config, scoring_rule=scoring_rule))
    print(f"Direct inference: {scoring_rule}, S={DIRECT_SUMMARY_DIM}")
    estimates = approximator.estimate(conditions=data_conditions(data), return_summaries=True)
    return {
        "id": np.asarray(datasets["ids"]),
        "candidate_models": np.asarray(ASSUMED_MODELS),
        "generating_model": np.full(num_datasets, datasets["generating_model"]),
        "well_specified": np.full(num_datasets, datasets["well_specified"], dtype=bool),
        "pmp": np.asarray(estimates["model_probs"], dtype=np.float64),
        "log_odds": np.asarray(estimates["log_odds"], dtype=np.float64),
        "summaries": np.asarray(estimates["_summaries"], dtype=np.float32),
    }

def load_dataset(split, source):
    if split == "benchmark":
        return load_benchmark_dataset(source)
    if split == "simulated":
        return load_simulated_dataset(source)
    raise ValueError("split must be 'benchmark' or 'simulated'")

def run_indirect(*, split, source, summary_multiplier, num_samples=2048, seed=2025, batch_size=None, overwrite=False):
    datasets = load_dataset(split, source)
    results = evaluate_indirect(
        datasets, summary_multiplier=summary_multiplier, num_samples=num_samples, seed=seed, batch_size=batch_size
    )
    path = inference_path(method="indirect", split=split, source=source, summary_multiplier=summary_multiplier)
    save_inference(results, path, overwrite=overwrite)
    print(f"Saved: {path}")

def run_direct(*, split, source, scoring_rule, overwrite=False):
    datasets = load_dataset(split, source)
    results = evaluate_direct(datasets, scoring_rule=scoring_rule)
    path = inference_path(method="direct", split=split, source=source, scoring_rule=scoring_rule)
    save_inference(results, path, overwrite=overwrite)
    print(f"Saved: {path}")

"""Compute analytical gold-standard results for Gaussian datasets."""

from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .analytic import GaussianAnalytical
from ..config import ASSUMED_MODELS, DATASET_DIR, MODEL_SPECS, RESULT_DIR
from ..datasets.datasets import load_datasets


GOLD_DIR = RESULT_DIR / "gold"


def compute_pmp(logml: np.ndarray, model_prior: np.ndarray | None = None) -> np.ndarray:
    """Convert log marginal likelihoods to posterior model probabilities."""

    num_models = logml.shape[1]

    if model_prior is None:
        model_prior = np.full(num_models, 1.0 / num_models)

    model_prior = np.asarray(model_prior, dtype=np.float64)

    if model_prior.shape != (num_models,):
        raise ValueError(f"model_prior must have shape ({num_models},)")

    if not np.isclose(model_prior.sum(), 1.0):
        raise ValueError("model_prior must sum to 1.")

    log_joint = logml + np.log(model_prior)[None, :]

    log_normalizer = logsumexp(log_joint, axis=1, keepdims=True)

    return np.exp(log_joint - log_normalizer)


def compute_gold(datasets: dict[str, np.ndarray], *, num_samples: int, seed: int = 2025) -> dict[str, np.ndarray]:
    """Compute analytical posterior samples, logML, and PMP."""

    x = np.asarray(datasets["x"], dtype=np.float64)

    ids = np.asarray(datasets["id"])

    source_model = np.asarray(datasets["source_model"])

    num_datasets, num_obs, num_dims = x.shape
    num_models = len(ASSUMED_MODELS)

    logml = np.empty((num_datasets, num_models), dtype=np.float64)

    posterior_samples = np.empty((num_datasets, num_models, num_samples, num_dims), dtype=np.float64)

    for model_index, model in enumerate(ASSUMED_MODELS):
        spec = MODEL_SPECS[model]

        for dataset_index in range(num_datasets):
            rng = np.random.default_rng(np.random.SeedSequence([seed, model_index, int(ids[dataset_index])]))

            analytical = GaussianAnalytical(
                obs_data=x[dataset_index],
                mu_prior_mean=spec["mu_prior_mean"],
                mu_prior_std=spec["mu_prior_std"],
                num_dims=num_dims,
                num_obs=num_obs,
                num_samples=num_samples,
                likelihood_std=spec["likelihood_std"],
                rng=rng,
            )

            posterior_samples[dataset_index, model_index] = analytical.analytical_posterior()

            logml[dataset_index, model_index] = analytical.log_marginal_analytical()

    pmp = compute_pmp(logml)

    return {
        "id": ids,
        "source_model": source_model,
        "candidate_models": np.asarray(ASSUMED_MODELS),
        "logml": logml,
        "pmp": pmp,
        "posterior_samples": posterior_samples,
    }


def save_gold(results: dict[str, np.ndarray], path: str | Path, *, overwrite: bool = False) -> None:
    """Save gold-standard results."""

    path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(f"Gold result already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(path, **results)


def compute_gold_file(
    *, split: str, source_model: str, num_obs: int, num_samples: int, seed: int = 2025, overwrite: bool = False
) -> None:
    """Compute gold results for one stored dataset file."""

    dataset_path = DATASET_DIR / f"n{num_obs}" / split / f"{source_model}.npz"

    output_path = GOLD_DIR / f"n{num_obs}" / split / f"{source_model}.npz"

    datasets = load_datasets(dataset_path)

    results = compute_gold(datasets, num_samples=num_samples, seed=seed)

    save_gold(results, output_path, overwrite=overwrite)

    print(f"Saved gold results: {split}, {source_model}, N={num_obs}")

"""Generate fixed datasets for the Gaussian experiments."""

import numpy as np

from ..config import (
    ASSUMED_MODELS,
    SOURCE_MODELS,
    MODEL_SPECS,
    DATASET_DIR,
)
from .datasets import (
    GaussianDatasetGenerator,
    save_datasets,
)


SPLIT_IDS = {
    "reference": 1,
    "calibration": 2,
    "benchmark": 3,
    "simulated": 4,
}


def get_rng(
    split: str,
    model: str,
    num_obs: int,
    seed: int,
):
    """Create a reproducible RNG for one dataset split and model."""

    if split not in SPLIT_IDS:
        raise ValueError(
            f"Unknown split: {split}"
        )

    model_id = int(model.removeprefix("m"))

    seed_sequence = np.random.SeedSequence(
        [
            seed,
            SPLIT_IDS[split],
            model_id,
            num_obs,
        ]
    )

    return np.random.default_rng(
        seed_sequence
    )


def generate_split(
    split: str,
    models,
    *,
    num_datasets: int,
    num_dims: int,
    num_obs: int,
    seed: int = 2025,
    overwrite: bool = False,
):
    """Generate and save one dataset split."""

    for model in models:
        spec = MODEL_SPECS[model]

        rng = get_rng(
            split=split,
            model=model,
            num_obs=num_obs,
            seed=seed,
        )

        generator = GaussianDatasetGenerator(
            mu_prior_mean=spec["mu_prior_mean"],
            mu_prior_std=spec["mu_prior_std"],
            num_dims=num_dims,
            num_obs=num_obs,
            likelihood_std=spec["likelihood_std"],
            rng=rng,
        )

        datasets = generator.simulate(
            num_datasets=num_datasets,
            source_model=model,
        )

        path = (
            DATASET_DIR
            / f"n{num_obs}"
            / split
            / f"{model}.npz"
        )

        save_datasets(
            datasets,
            path,
            overwrite=overwrite,
        )

        print(
            f"Saved {split}: "
            f"{model}, "
            f"N={num_obs}, "
            f"B={num_datasets}"
        )


def generate_all(
    *,
    num_reference: int,
    num_calibration: int,
    num_benchmark: int,
    num_simulated: int,
    num_dims: int = 20,
    num_obs_values=(10, 100),
    seed: int = 2025,
    overwrite: bool = False,
):
    """Generate all fixed datasets."""

    for num_obs in num_obs_values:

        generate_split(
            split="reference",
            models=ASSUMED_MODELS,
            num_datasets=num_reference,
            num_dims=num_dims,
            num_obs=num_obs,
            seed=seed,
            overwrite=overwrite,
        )

        generate_split(
            split="calibration",
            models=ASSUMED_MODELS,
            num_datasets=num_calibration,
            num_dims=num_dims,
            num_obs=num_obs,
            seed=seed,
            overwrite=overwrite,
        )

        generate_split(
            split="benchmark",
            models=ASSUMED_MODELS,
            num_datasets=num_benchmark,
            num_dims=num_dims,
            num_obs=num_obs,
            seed=seed,
            overwrite=overwrite,
        )

        generate_split(
            split="simulated",
            models=SOURCE_MODELS,
            num_datasets=num_simulated,
            num_dims=num_dims,
            num_obs=num_obs,
            seed=seed,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    generate_all(
        num_reference=2000,
        num_calibration=1000,
        num_benchmark=100,
        num_simulated=50,
        overwrite=False,
    )
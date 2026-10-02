"""Train Gaussian indirect-network ablations."""

import itertools

from ..config import ASSUMED_MODELS
from .config import TrainingConfig
from .indirect import train_one


NUM_OBS = 10
SUMMARY_DIMS = (40, 80)

VARIANTS = {
    "seed_2026": {
        "seed": 2026,
        "learning_rate_schedule": "cosine",
        "standardize": "all",
    },
    "fixed_lr": {
        "seed": 2025,
        "learning_rate_schedule": "constant",
        "standardize": "all",
    },
    "no_standardize": {
        "seed": 2025,
        "learning_rate_schedule": "cosine",
        "standardize": None,
    },
}


def train_variant(
    variant: str,
    *,
    overwrite=False,
):
    settings = VARIANTS[variant]

    for model, summary_dim in itertools.product(
        ASSUMED_MODELS,
        SUMMARY_DIMS,
    ):
        config = TrainingConfig(
            num_dims=20,
            num_obs=NUM_OBS,
            summary_dim=summary_dim,
            epochs=256,
            batch_size=64,
            num_batches=128,
            learning_rate=1e-4,
            summary_base_distribution="normal",
            **settings,
        )

        train_one(
            model=model,
            config=config,
            variant=variant,
            overwrite=overwrite,
        )


def train_all_ablation(*, overwrite=False):
    for variant in VARIANTS:
        train_variant(
            variant,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    train_all_ablation(overwrite=False)
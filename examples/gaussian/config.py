"""Shared configuration for the Gaussian case study."""

from pathlib import Path


# Paths
BASE_DIR = Path(__file__).resolve().parent

NETWORK_DIR = BASE_DIR / "networks"
RESULT_DIR = BASE_DIR / "results"
DATASET_DIR = RESULT_DIR / "datasets"


# Gaussian model specifications
MODEL_SPECS = {
    "m1": {
        "mu_prior_mean": 0.0,
        "mu_prior_std": 1.0,
        "likelihood_std": 1.0,
    },
    "m2": {
        "mu_prior_mean": 3.0,
        "mu_prior_std": 1.0,
        "likelihood_std": 1.0,
    },
    "m3": {
        "mu_prior_mean": 0.0,
        "mu_prior_std": 1.0,
        "likelihood_std": 3.0,
    },
    "m4": {
        "mu_prior_mean": 0.1,
        "mu_prior_std": 1.0,
        "likelihood_std": 1.0,
    },
    "m5": {
        "mu_prior_mean": 1.5,
        "mu_prior_std": 1.0,
        "likelihood_std": 1.0,
    },
    "m6": {
        "mu_prior_mean": 5.0,
        "mu_prior_std": 1.0,
        "likelihood_std": 1.0,
    },
    "m7": {
        "mu_prior_mean": 0.0,
        "mu_prior_std": 1.0,
        "likelihood_std": 5.0,
    },
    "m8": {
        "mu_prior_mean": 0.0,
        "mu_prior_std": 3.0,
        "likelihood_std": 1.0,
    },
    "m9": {
        "mu_prior_mean": 0.0,
        "mu_prior_std": 1.0,
        "likelihood_std": 0.1,
    },
    "m10": {
        "mu_prior_mean": 0.0,
        "mu_prior_std": 1.0,
        "likelihood_std": 0.01,
    },
    "m11": {
        "mu_prior_mean": 0.0,
        "mu_prior_std": 0.1,
        "likelihood_std": 1.0,
    },
    "m12": {
        "mu_prior_mean": 0.0,
        "mu_prior_std": 0.01,
        "likelihood_std": 1.0,
    },
}


# Models used for inference / model comparison
ASSUMED_MODELS = ("m1", "m2", "m3", "m4")

# Models used to generate evaluation datasets
SOURCE_MODELS = tuple(MODEL_SPECS)

def get_result_dir(
    variant: str = "baseline",
):
    if variant == "baseline":
        return RESULT_DIR

    return RESULT_DIR / "ablation" / variant
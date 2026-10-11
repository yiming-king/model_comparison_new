"""Shared configuration for the diffusion case study."""

from pathlib import Path

import numpy as np


# Paths
BASE_DIR = Path(__file__).resolve().parent

NETWORK_DIR = BASE_DIR / "networks"
RESULT_DIR = BASE_DIR / "results"

DATASET_DIR = RESULT_DIR / "datasets"

REFERENCE_DATASET_DIR = DATASET_DIR / "reference"
CALIBRATION_DATASET_DIR = DATASET_DIR / "calibration"

BENCHMARK_DATASET_DIR = DATASET_DIR / "benchmark"
SIMULATED_DATASET_DIR = DATASET_DIR / "simulated"
EMPIRICAL_DATASET_DIR = BASE_DIR / "dataset" / "json" / "empirical"

GOLD_DIR = RESULT_DIR / "gold"
BENCHMARK_GOLD_DIR = GOLD_DIR / "benchmark"
SIMULATED_GOLD_DIR = GOLD_DIR / "simulated"
EMPIRICAL_GOLD_DIR = BASE_DIR / "stan" / "results_4_models" / "empirical"


# Indirect-network ablations. The baseline keeps all existing output paths.
INDIRECT_VARIANTS = {
    "baseline": {"standardize": "all"},
    "inference_only": {"standardize": "inference_variables"},
}


def normalize_variant(variant: str = "baseline") -> str:
    variant = "baseline" if variant == "default" else variant
    if variant not in INDIRECT_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {tuple(INDIRECT_VARIANTS)} or 'default'.")
    return variant


def get_variant_dir(root, variant: str = "baseline"):
    variant = normalize_variant(variant)
    root = Path(root)
    return root if variant == "baseline" else root / "ablation" / variant


def get_result_dir(variant: str = "baseline", *, result_dir=RESULT_DIR):
    """Select derived outputs; dataset and gold directories stay shared."""
    return get_variant_dir(result_dir, variant)


# Assumed RDM models
ASSUMED_MODELS = ("m0", "m1", "m2", "m3")

N_ALPHA = {"m0": 1, "m1": 2, "m2": 2, "m3": 4}

PARAM_DIMS = {model: n_alpha + 3 for model, n_alpha in N_ALPHA.items()}


# Fixed experimental design
NUM_TRIALS = 384

CONDITIONS = np.repeat(np.array([1, 0, 1, 0], dtype=np.int32), 96)


# Fixed evaluation datasets
SIMULATED_DATASETS = (
    "simulated_from_m0",
    "simulated_from_m1",
    "simulated_from_m2",
    "simulated_from_m3",
    "m3_fast_30",
    "m3_slow_30",
    "m3_fast_slow_30",
)

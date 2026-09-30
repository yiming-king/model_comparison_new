"""Shared configuration for the diffusion case study."""

from pathlib import Path

import numpy as np


# Paths
BASE_DIR = Path(__file__).resolve().parent

NETWORK_DIR = BASE_DIR / "networks"
RESULT_DIR = BASE_DIR / "results"

DATASET_DIR = RESULT_DIR / "datasets"
BENCHMARK_DATASET_DIR = DATASET_DIR / "benchmark"
SIMULATED_DATASET_DIR = DATASET_DIR / "simulated"

GOLD_DIR = RESULT_DIR / "gold"
BENCHMARK_GOLD_DIR = GOLD_DIR / "benchmark"
SIMULATED_GOLD_DIR = GOLD_DIR / "simulated"


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

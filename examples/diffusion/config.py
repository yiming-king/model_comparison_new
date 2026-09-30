"""Shared paths and fixed dataset names for the diffusion case study."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

DATASET_DIR = RESULT_DIR / "datasets"
BENCHMARK_DATASET_DIR = DATASET_DIR / "benchmark"
SIMULATED_DATASET_DIR = DATASET_DIR / "simulated"

GOLD_DIR = RESULT_DIR / "gold"
BENCHMARK_GOLD_DIR = GOLD_DIR / "benchmark"
SIMULATED_GOLD_DIR = GOLD_DIR / "simulated"

ASSUMED_MODELS = ("m0", "m1", "m2", "m3")
SIMULATED_DATASETS = (
    "simulated_from_m0",
    "simulated_from_m1",
    "simulated_from_m2",
    "simulated_from_m3",
    "m3_fast_30",
    "m3_slow_30",
    "m3_fast_slow_30",
)

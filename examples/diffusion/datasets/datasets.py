"""Read fixed JSON datasets without generating or changing observations."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ASSUMED_MODELS, BENCHMARK_DATASET_DIR, SIMULATED_DATASET_DIR, SIMULATED_DATASETS

_SIMULATED_INFO = {
    "simulated_from_m0": ("m0", True),
    "simulated_from_m1": ("m1", True),
    "simulated_from_m2": ("m2", True),
    "simulated_from_m3": ("m3", True),
    "m3_fast_30": ("m3", False),
    "m3_slow_30": ("m3", False),
    "m3_fast_slow_30": ("m3", False),
}

def _id_sort_key(path: Path) -> tuple[str, int, str]:
    match = re.fullmatch(r"(.+?)(\d+)", path.stem)
    if match is None:
        return path.stem, -1, path.stem
    return match.group(1), int(match.group(2)), path.stem

def load_json_directory(path: str | Path) -> dict:
    """Load JSON observations as (dataset, trial, [rt, condition])."""
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    files = sorted(directory.glob("*.json"), key=_id_sort_key)
    if not files:
        raise ValueError(f"No JSON datasets found in {directory}")

    data = []
    n_trials = None
    for file in files:
        with file.open() as stream:
            record = json.load(stream)
        rt = np.asarray(record["rt"], dtype=np.float64)
        condition = np.asarray(record["condition"], dtype=np.float64)
        if rt.ndim != 1 or condition.ndim != 1 or len(rt) != len(condition) or len(rt) != record["N"]:
            raise ValueError(f"Invalid trial dimensions in {file}")
        if n_trials is None:
            n_trials = len(rt)
        elif len(rt) != n_trials:
            raise ValueError(f"Inconsistent trial count in {file}")
        data.append(np.stack((rt, condition), axis=-1))

    return {"data": np.stack(data), "ids": [file.stem for file in files]}

def _load_true_parameters(directory: Path, ids: list[str]) -> pd.DataFrame:
    parameters = pd.read_csv(directory / "true_parameters.csv", dtype={"id": str})
    parameter_ids = parameters["id"].tolist()
    if len(parameter_ids) != len(ids) or set(parameter_ids) != set(ids):
        raise ValueError(f"Parameter IDs do not match JSON filenames in {directory}")
    return parameters

def load_simulated_dataset(name: str) -> dict:
    """Load one fixed simulated group, including its contamination status."""
    if name not in SIMULATED_DATASETS:
        raise ValueError(f"Unknown simulated dataset: {name}")
    directory = SIMULATED_DATASET_DIR / name
    loaded = load_json_directory(directory)
    generating_model, well_specified = _SIMULATED_INFO[name]
    return {
        **loaded,
        "true_parameters": _load_true_parameters(directory, loaded["ids"]),
        "source": directory,
        "generating_model": generating_model,
        "well_specified": well_specified,
    }

def load_benchmark_dataset(model: str) -> dict:
    """Load one generating model's fixed benchmark datasets."""
    if model not in ASSUMED_MODELS:
        raise ValueError(f"Unknown benchmark model: {model}")
    directory = BENCHMARK_DATASET_DIR / model
    loaded = load_json_directory(directory)
    manifest = pd.read_csv(BENCHMARK_DATASET_DIR / "dataset_manifest.csv", dtype={"generating_model": str, "id": str})
    model_manifest = manifest.loc[manifest["generating_model"] == model].copy()
    manifest_ids = model_manifest["id"].tolist()
    if len(manifest_ids) != len(loaded["ids"]) or set(manifest_ids) != set(loaded["ids"]):
        raise ValueError(f"Benchmark manifest IDs do not match JSON filenames for {model}")
    return {
        **loaded,
        "true_parameters": _load_true_parameters(directory, loaded["ids"]),
        "manifest": model_manifest,
        "source": directory,
        "generating_model": model,
        "well_specified": True,
    }

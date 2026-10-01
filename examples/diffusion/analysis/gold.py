"""Load Stan and bridge-sampling gold standards for the diffusion case study."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from ..config import ASSUMED_MODELS, BENCHMARK_GOLD_DIR, N_ALPHA, SIMULATED_GOLD_DIR
from ..datasets import load_benchmark_dataset, load_simulated_dataset

def gold_root(split, source):
    if split == "benchmark":
        return BENCHMARK_GOLD_DIR / source
    if split == "simulated":
        return SIMULATED_GOLD_DIR / source
    raise ValueError("split must be 'benchmark' or 'simulated'")

def parameter_columns(model):
    return [f"alpha_{i}" for i in range(N_ALPHA[model])] + ["nu_0", "nu_1", "tau"]

def load_dataset(split, source):
    if split == "benchmark":
        return load_benchmark_dataset(source)
    if split == "simulated":
        return load_simulated_dataset(source)
    raise ValueError("split must be 'benchmark' or 'simulated'")

def load_bridge_table(split, source, candidate_model):
    path = gold_root(split, source) / candidate_model / "bridgesampling.csv"
    if not path.exists():
        raise FileNotFoundError(f"Bridge-sampling result not found: {path}")
    table = pd.read_csv(path, dtype={"id": str})
    required = {"id", "estimate"}
    if not required.issubset(table.columns):
        raise ValueError(f"Invalid bridge-sampling table: {path}")
    return table

def load_convergence(split, source, candidate_model, ids):
    model_dir = gold_root(split, source) / candidate_model
    bridge = load_bridge_table(split, source, candidate_model)
    if "converged" in bridge.columns:
        table = bridge[["id", "converged"]].copy()
    else:
        path = model_dir / "convergence_diagnostics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Convergence diagnostics not found: {path}")
        table = pd.read_csv(path, dtype={"id": str})[["id", "converged"]]
    table = table.set_index("id", verify_integrity=True)
    missing = set(ids) - set(table.index)
    if missing:
        raise ValueError(f"Missing convergence results for {candidate_model}: {sorted(missing)}")
    values = table.loc[ids, "converged"]
    if values.dtype != bool:
        values = values.astype(str).str.lower().isin(["true", "1", "t"])
    return values.to_numpy(dtype=bool)

def load_posterior_draws(split, source, candidate_model, dataset_id):
    path = gold_root(split, source) / candidate_model / "posterior_draws" / f"{dataset_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stan posterior draws not found: {path}")
    frame = pd.read_csv(path)
    columns = parameter_columns(candidate_model)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing posterior columns in {path}: {missing}")
    return frame[columns].to_numpy(dtype=np.float32)

def load_gold(split, source):
    datasets = load_dataset(split, source)
    ids = list(datasets["ids"])
    num_datasets = len(ids)
    num_models = len(ASSUMED_MODELS)
    logml = np.empty((num_datasets, num_models), dtype=np.float64)
    bridge_sd = np.empty((num_datasets, num_models), dtype=np.float64)
    converged = np.empty((num_datasets, num_models), dtype=bool)
    for model_index, model in enumerate(ASSUMED_MODELS):
        table = load_bridge_table(split, source, model).set_index("id", verify_integrity=True)
        missing = set(ids) - set(table.index)
        if missing:
            raise ValueError(f"Missing bridge results for {model}: {sorted(missing)}")
        logml[:, model_index] = table.loc[ids, "estimate"].to_numpy(dtype=np.float64)
        if "sd" in table.columns:
            bridge_sd[:, model_index] = table.loc[ids, "sd"].to_numpy(dtype=np.float64)
        else:
            bridge_sd[:, model_index] = np.nan
        converged[:, model_index] = load_convergence(split, source, model, ids)
    pmp = softmax(logml, axis=-1)
    return {
        "id": np.asarray(ids),
        "candidate_models": np.asarray(ASSUMED_MODELS),
        "generating_model": np.full(num_datasets, datasets["generating_model"]),
        "well_specified": np.full(num_datasets, datasets["well_specified"], dtype=bool),
        "logml": logml,
        "bridge_sd": bridge_sd,
        "pmp": pmp,
        "converged": converged,
        "all_converged": np.all(converged, axis=1),
    }

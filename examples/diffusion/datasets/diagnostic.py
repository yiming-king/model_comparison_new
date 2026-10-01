"""Generate fixed reference and calibration datasets for RDM diagnostics."""

from pathlib import Path

import numpy as np

from ..config import ASSUMED_MODELS, CALIBRATION_DATASET_DIR, NUM_TRIALS, REFERENCE_DATASET_DIR
from ..simulators import SIMULATORS_NO_PARAMS

SPLIT_IDS = {"reference": 1, "calibration": 2}

def dataset_path(split, model):
    """Return the path for one diagnostic dataset bank."""
    if split == "reference":
        root = REFERENCE_DATASET_DIR
    elif split == "calibration":
        root = CALIBRATION_DATASET_DIR
    else:
        raise ValueError("split must be 'reference' or 'calibration'")
    return root / f"{model}.npz"

def simulation_seed(*, split, model, seed=2025):
    """Create one reproducible seed for a split/model pair."""
    if split not in SPLIT_IDS:
        raise ValueError(f"Unknown split: {split}")
    if model not in ASSUMED_MODELS:
        raise ValueError(f"Unknown model: {model}")
    model_id = int(model.removeprefix("m"))
    sequence = np.random.SeedSequence([seed, SPLIT_IDS[split], model_id, NUM_TRIALS])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])

def simulate_datasets(model, *, num_datasets, seed):
    """
    Simulate fixed prior-predictive datasets.

    Returns:
        data with shape:
            (num_datasets, NUM_TRIALS, 2)

        where the last dimension is:
            [rt, condition]
    """
    simulator = SIMULATORS_NO_PARAMS[model]
    # RDM simulators currently use NumPy's global RNG.
    # Save and restore its state so this function does not
    # change randomness elsewhere in the program.
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        samples = simulator.sample(num_datasets)
    finally:
        np.random.set_state(state)
    data = np.stack([samples["rt"], samples["conditions"]], axis=-1).astype(np.float32)
    expected_shape = (num_datasets, NUM_TRIALS, 2)
    if data.shape != expected_shape:
        raise ValueError(f"Expected data shape {expected_shape}, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values generated for {model}")
    return data

def save_datasets(data, path, *, model, split, seed, overwrite=False):
    """Save one fixed diagnostic dataset bank."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Dataset already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray([f"{split}_{model}_{i:04d}" for i in range(len(data))])
    np.savez_compressed(
        path,
        data=np.asarray(data, dtype=np.float32),
        ids=ids,
        generating_model=np.asarray(model),
        well_specified=np.asarray(True),
        seed=np.asarray(seed, dtype=np.uint32),
    )

def load_diagnostic_dataset(split, model):
    """Load one fixed reference or calibration dataset bank."""
    path = dataset_path(split, model)
    if not path.exists():
        raise FileNotFoundError(f"Diagnostic dataset not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def generate_split(split, *, num_datasets, seed=2025, overwrite=False):
    """Generate one split for all four assumed models."""
    for model in ASSUMED_MODELS:
        model_seed = simulation_seed(split=split, model=model, seed=seed)
        data = simulate_datasets(model, num_datasets=num_datasets, seed=model_seed)
        path = dataset_path(split, model)
        save_datasets(data, path, model=model, split=split, seed=model_seed, overwrite=overwrite)
        print(f"Saved {split}: {model}, B={num_datasets}, N={NUM_TRIALS}")

def generate_all(*, num_reference=2000, num_calibration=2000, seed=2025, overwrite=False):
    """Generate all diagnostic reference datasets."""
    generate_split("reference", num_datasets=num_reference, seed=seed, overwrite=overwrite)
    generate_split("calibration", num_datasets=num_calibration, seed=seed, overwrite=overwrite)

if __name__ == "__main__":
    generate_all()

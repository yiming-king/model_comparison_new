import argparse
import json
import os
from pathlib import Path

try:
    from examples.diffusion.config import SIMULATED_DATASET_DIR
    from examples.diffusion.dataset.dataset import wagenmakers
    from simulators import SIMULATORS
except ImportError:
    from ..config import SIMULATED_DATASET_DIR
    from . import wagenmakers
    from ..simulators import SIMULATORS

import pandas as pd
import numpy as np

N_SIM = 17
BASE_DIR = SIMULATED_DATASET_DIR

N = wagenmakers.n_trials

def sample_seeded_dataset(simulator, seed: int) -> dict[str, np.ndarray]:
    """Draw one dataset without leaking changes to NumPy's global RNG state."""
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return simulator.sample(1)
    finally:
        np.random.set_state(state)

def save_seeded_datasets(
    simulator, output_dir: str | Path, seeds: list[int], *, overwrite: bool = False
) -> pd.DataFrame:
    """Save independently seeded simulator draws and their reproducibility manifest."""
    output_dir = Path(output_dir)
    owned_files = [*output_dir.glob("s*.json"), output_dir / "true_parameters.csv"]
    if any(path.exists() for path in owned_files) and not overwrite:
        raise FileExistsError(
            f"Calibration datasets already exist in {output_dir}; pass --overwrite to regenerate them."
        )
    if overwrite:
        for path in owned_files:
            if path.exists():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, seed in enumerate(seeds):
        data = sample_seeded_dataset(simulator, int(seed))
        dataset_id = f"s{index:03d}"
        record = {
            "rt": [float(value) for value in data["rt"][0]],
            "condition": [int(value) for value in data["conditions"][0]],
            "N": N,
        }
        with (output_dir / f"{dataset_id}.json").open("w") as stream:
            json.dump(record, stream)

        row = {"id": dataset_id, "dataset_seed": int(seed)}
        for key in ("alpha", "nu", "tau"):
            for parameter_index, value in enumerate(data[key][0].reshape(-1)):
                row[f"{key}_{parameter_index}"] = float(value)
        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "true_parameters.csv", index=False)
    return manifest

def save_datasets(simulator, folder: str, n_sim: int = N_SIM, overwrite: bool = False):
    out_dir = os.path.join(BASE_DIR, folder)
    if os.path.exists(out_dir) and not overwrite:
        raise FileExistsError(f"{out_dir} already exists. Pass --overwrite to regenerate it.")
    os.makedirs(out_dir, exist_ok=True)

    data = simulator.sample(n_sim)
    parameter_rows = []

    for i in range(n_sim):
        record = dict(rt=[float(x) for x in data["rt"][i]], condition=[int(x) for x in data["conditions"][i]], N=N)
        path = os.path.join(out_dir, f"s{i}.json")
        with open(path, "w") as f:
            json.dump(record, f)

        row = {"id": f"s{i}"}
        for key in ("alpha", "nu", "tau"):
            values = data[key][i].reshape(-1)
            for j, value in enumerate(values):
                row[f"{key}_{j}"] = float(value)
        parameter_rows.append(row)

    pd.DataFrame(parameter_rows).to_csv(os.path.join(out_dir, "true_parameters.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--n-sim", type=int, default=N_SIM)
    parser.add_argument("--seed", type=int, default=2025)  # Random seed for reproducibility
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        import numpy as np

        np.random.seed(args.seed)

    selected = SIMULATORS.keys() if args.models == ["all"] else args.models
    for model in selected:
        save_datasets(SIMULATORS[model], folder=f"simulated_from_{model}", n_sim=args.n_sim, overwrite=args.overwrite)

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from examples.diffusion.config import SIMULATED_DATASET_DIR
    from examples.diffusion.dataset.dataset import wagenmakers
    from simulators import SIMULATORS
except ImportError:
    from ..config import SIMULATED_DATASET_DIR
    from . import wagenmakers
    from ..simulators import SIMULATORS


N_SIM = 17
SOURCE_MODEL = "m3"
BASE_DIR = SIMULATED_DATASET_DIR


def _replace_rt(
    rt: np.ndarray,
    conditions: np.ndarray,
    mask: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> None:
    for condition in np.unique(conditions):
        for sign in (-1.0, 1.0):
            group = (conditions == condition) & (np.sign(rt) == sign)
            selected = group & mask
            if not np.any(selected):
                continue

            magnitudes = np.abs(rt[group])
            if mode == "fast":
                low, high = 0.1, np.quantile(magnitudes, 0.10)
            elif mode == "slow":
                low, high = np.quantile(magnitudes, 0.75), 10.0
            else:
                raise ValueError(f"Unknown contamination mode: {mode}")

            low, high = float(low), float(max(high, low + 1e-6))
            rt[selected] = sign * rng.uniform(low, high, size=int(selected.sum()))


def contaminate_rt(
    rt: np.ndarray,
    conditions: np.ndarray,
    kind: str,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.asarray(rt, dtype=float).copy()
    n_trials = output.shape[0]

    if kind == "fast_slow":
        n_fast = int(round(n_trials * fraction / 2.0))
        n_slow = int(round(n_trials * fraction)) - n_fast
        indices = rng.permutation(n_trials)
        fast_mask = np.zeros(n_trials, dtype=bool)
        slow_mask = np.zeros(n_trials, dtype=bool)
        fast_mask[indices[:n_fast]] = True
        slow_mask[indices[n_fast : n_fast + n_slow]] = True
        _replace_rt(output, conditions, fast_mask, "fast", rng)
        _replace_rt(output, conditions, slow_mask, "slow", rng)
        return output

    n_replace = int(round(n_trials * fraction))
    mask = np.zeros(n_trials, dtype=bool)
    mask[rng.choice(n_trials, size=n_replace, replace=False)] = True
    _replace_rt(output, conditions, mask, kind, rng)
    return output


def save_contaminated_datasets(
    kind: str,
    fraction: float = 0.30,
    n_sim: int = N_SIM,
    seed: int = 2025,
    overwrite: bool = False,
) -> Path:
    suffix = int(round(100 * fraction))
    folder = f"{SOURCE_MODEL}_{kind}_{suffix}"
    out_dir = BASE_DIR / folder
    if out_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{out_dir} already exists. Pass --overwrite to regenerate it."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    numpy_state = np.random.get_state()
    try:
        np.random.seed(seed)
        data = SIMULATORS[SOURCE_MODEL].sample(n_sim)
    finally:
        np.random.set_state(numpy_state)

    parameter_rows = []

    for i in range(n_sim):
        rt = contaminate_rt(data["rt"][i], data["conditions"][i], kind, fraction, rng)
        record = {
            "rt": [float(x) for x in rt],
            "condition": [int(x) for x in data["conditions"][i]],
            "N": wagenmakers.n_trials,
        }
        with (out_dir / f"s{i}.json").open("w") as f:
            json.dump(record, f)

        row = {
            "id": f"s{i}",
            "source_model": SOURCE_MODEL,
            "contamination": kind,
            "fraction": fraction,
        }
        for key in ("alpha", "nu", "tau"):
            for j, value in enumerate(data[key][i].reshape(-1)):
                row[f"{key}_{j}"] = float(value)
        parameter_rows.append(row)

    pd.DataFrame(parameter_rows).to_csv(out_dir / "true_parameters.csv", index=False)
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", default=["fast", "slow", "fast_slow"])
    parser.add_argument("--fraction", type=float, default=0.30)
    parser.add_argument("--n-sim", type=int, default=N_SIM)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for offset, kind in enumerate(args.kinds):
        path = save_contaminated_datasets(
            kind=kind,
            fraction=args.fraction,
            n_sim=args.n_sim,
            seed=args.seed + offset,
            overwrite=args.overwrite,
        )
        print(path)

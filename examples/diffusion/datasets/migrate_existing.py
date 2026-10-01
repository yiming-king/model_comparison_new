"""Copy and check the fixed RDM evaluation data from Model-comparison.

Run with ``python -m examples.diffusion.datasets.migrate_existing --old-repo PATH``.
Use ``--check-only`` to validate an existing results directory without copying.
"""

import argparse
import filecmp
from pathlib import Path
from shutil import copy2

import numpy as np
import pandas as pd

from ..config import ASSUMED_MODELS, NUM_TRIALS, RESULT_DIR, SIMULATED_DATASETS
from .datasets import load_benchmark_dataset, load_simulated_dataset


def copy_missing(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
    for file in files:
        target = destination if source.is_file() else destination / file.relative_to(source)
        if target.exists():
            if not filecmp.cmp(file, target, shallow=False):
                raise ValueError(f"Existing file differs from legacy source: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(file, target)


def migrate(old_repo: Path) -> None:
    old = old_repo / "benchmark/examples/diffusion"
    datasets = RESULT_DIR / "datasets"
    gold = RESULT_DIR / "gold"
    copy_missing(old / "calibration_reference_100/dataset_manifest.csv", datasets / "benchmark/dataset_manifest.csv")
    for model in ASSUMED_MODELS:
        copy_missing(old / "calibration_reference_100/datasets" / model, datasets / "benchmark" / model)
        copy_missing(old / "calibration_reference_100/mcmc" / model, gold / "benchmark" / model)
    for source in SIMULATED_DATASETS:
        copy_missing(old / "dataset/json" / source, datasets / "simulated" / source)
        copy_missing(old / "stan/results_4_models" / source, gold / "simulated" / source)


def csv_ids(path: Path) -> set[str]:
    table = pd.read_csv(path, dtype={"id": str})
    if "id" not in table or table["id"].isna().any() or table["id"].duplicated().any():
        raise ValueError(f"Missing or duplicate IDs in {path}")
    return set(table["id"])


def require_same_ids(expected: set[str], actual: set[str], label: str, errors: list[str]) -> None:
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        errors.append(f"{label}: missing {len(missing)} {missing[:3]}, extra {len(extra)} {extra[:3]}")


def check_observations(loaded: dict, label: str, expected_count: int) -> set[str]:
    data = loaded["data"]
    ids = loaded["ids"]
    if len(ids) != expected_count or data.shape != (expected_count, NUM_TRIALS, 2):
        raise ValueError(f"{label}: expected {expected_count} datasets of shape ({NUM_TRIALS}, 2), got {data.shape}")
    if not np.isfinite(data).all() or not np.isin(data[..., 1], [0, 1]).all():
        raise ValueError(f"{label}: nonfinite observation or condition outside {{0, 1}}")
    return set(ids)


def check() -> None:
    datasets = RESULT_DIR / "datasets"
    gold = RESULT_DIR / "gold"
    manifest = pd.read_csv(datasets / "benchmark/dataset_manifest.csv", dtype={"generating_model": str, "id": str})
    if manifest[["generating_model", "id"]].isna().any().any() or manifest.duplicated(["generating_model", "id"]).any():
        raise ValueError("Benchmark manifest has missing or duplicate model/ID pairs")

    errors: list[str] = []
    for split, sources, expected_count in (("benchmark", ASSUMED_MODELS, 100), ("simulated", SIMULATED_DATASETS, 17)):
        for source in sources:
            loaded = load_benchmark_dataset(source) if split == "benchmark" else load_simulated_dataset(source)
            label = f"{split}/{source}"
            ids = check_observations(loaded, label, expected_count)
            directory = datasets / split / source
            require_same_ids(ids, csv_ids(directory / "true_parameters.csv"), f"{label} parameters", errors)
            if split == "benchmark":
                rows = manifest.loc[manifest["generating_model"] == source]
                require_same_ids(ids, set(rows["id"]), f"{label} manifest", errors)
            print(f"{label}: {len(ids)} JSON datasets, shape {loaded['data'].shape}")

            for candidate in ASSUMED_MODELS:
                model_dir = gold / split / source / candidate
                bridge = csv_ids(model_dir / "bridgesampling.csv")
                draws_dir = model_dir / "posterior_draws"
                optional_draws = split == "benchmark" and candidate != source
                if not draws_dir.is_dir() and not optional_draws:
                    raise FileNotFoundError(draws_dir)
                draws = {p.stem for p in draws_dir.glob("*.csv")} if draws_dir.is_dir() else set()
                require_same_ids(ids, bridge, f"{label}/{candidate} bridge", errors)
                if optional_draws:
                    # Legacy calibration only used matching-model draws for MMD.
                    if draws - ids:
                        errors.append(f"{label}/{candidate} posterior draws have unknown IDs: {sorted(draws - ids)}")
                else:
                    require_same_ids(ids, draws, f"{label}/{candidate} posterior draws", errors)
                diagnostics = model_dir / "convergence_diagnostics.csv"
                if diagnostics.exists():
                    require_same_ids(ids, csv_ids(diagnostics), f"{label}/{candidate} convergence", errors)

    if errors:
        raise ValueError("Consistency checks failed:\n" + "\n".join(errors))
    print("All required dataset and gold ID checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-repo", type=Path, help="Path to the old Model-comparison repository")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only:
        if args.old_repo is None:
            parser.error("--old-repo is required unless --check-only is set")
        migrate(args.old_repo)
    check()


if __name__ == "__main__":
    main()

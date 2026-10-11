"""Join saved RDM inference, gold standards, errors, and diagnostics.

This module does not run neural networks or recompute diagnostics. It combines
already-saved RDM results into plotting tables.

Expected inputs (all relative to examples/diffusion/results unless noted):

- inference/n384/<split>/<run>/<source>.npz
- metrics/n384/<split>/<run>/<source>.npz
- thresholds/n384/<run>.npz
- diagnostics/n384/<run>/thresholds.npz
- diagnostics/n384/<run>/<split>/<source>.npz
- gold/{benchmark,simulated}/<source>/<candidate>/bridgesampling.csv
- ../stan/results_4_models/empirical/<candidate>/bridgesampling.csv

The default comparison contains simulated and empirical datasets only.
Benchmark datasets are used for calibration and can be added later once their
saved diagnostic scores are available.
"""

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata

from ..config import (
    ASSUMED_MODELS,
    BASE_DIR,
    NUM_TRIALS,
    PARAM_DIMS,
    RESULT_DIR,
    SIMULATED_DATASETS,
)

DIAGNOSTICS = ("l2", "linf", "density", "mmd")
DIRECT_SUMMARY_DIM = 12
SCORING_RULES = ("cross_entropy", "logistic", "exponential")

KEYS = ["dataset_type", "source_model", "id", "candidate_model"]
DATASET_KEYS = KEYS[:-1]

ERROR_SPECS = {
    "posterior_mmd": (
        "posterior_mmd",
        "posterior_error_lower",
        "posterior_error_upper",
    ),
    "logml": (
        "signed_logml_error",
        "logml_error_lower",
        "logml_error_upper",
    ),
    "pmp": (
        "signed_pmp_error",
        "pmp_error_lower",
        "pmp_error_upper",
    ),
}

# fit_4_models.R explicitly keeps empirical Stan output here.
EMPIRICAL_GOLD_DIR = BASE_DIR / "stan" / "results_4_models" / "empirical"


def run_name(*, method, summary_multiplier=None, scoring_rule=None):
    """Return the directory name for one saved RDM analysis run."""
    if method == "indirect":
        if summary_multiplier not in (1, 2, 4):
            raise ValueError("summary_multiplier must be 1, 2, or 4")
        return f"indirect_s{summary_multiplier}D"

    if method == "direct":
        if scoring_rule not in SCORING_RULES:
            raise ValueError(f"scoring_rule must be one of {SCORING_RULES}")
        return f"direct_{scoring_rule}_s{DIRECT_SUMMARY_DIM}"

    raise ValueError("method must be 'indirect' or 'direct'")


def comparison_paths(
    *, method, summary_multiplier=None, scoring_rule=None, result_dir=RESULT_DIR
):
    """Return the three comparison CSV paths for one run."""
    name = run_name(
        method=method,
        summary_multiplier=summary_multiplier,
        scoring_rule=scoring_rule,
    )
    root = Path(result_dir) / "comparison" / f"n{NUM_TRIALS}" / name
    return {
        "data": root / "data.csv",
        "detection": root / "detection.csv",
        "global_pmp_detection": root / "global_pmp_detection.csv",
    }


def _gold_directory(*, split, source, result_dir):
    """Return the directory containing per-candidate bridge tables."""
    if split == "empirical":
        return EMPIRICAL_GOLD_DIR
    return Path(result_dir) / "gold" / split / source


def input_paths(
    *,
    method,
    split,
    source,
    summary_multiplier=None,
    scoring_rule=None,
    result_dir=RESULT_DIR,
):
    """Return all saved inputs needed to build one comparison block."""
    if split not in ("simulated", "empirical", "benchmark"):
        raise ValueError("split must be simulated, empirical, or benchmark")

    name = run_name(
        method=method,
        summary_multiplier=summary_multiplier,
        scoring_rule=scoring_rule,
    )
    root = Path(result_dir)
    relative = Path(f"n{NUM_TRIALS}") / split / name / f"{source}.npz"

    return {
        "metrics": root / "metrics" / relative,
        "inference": root / "inference" / relative,
        "diagnostics": (
            root
            / "diagnostics"
            / f"n{NUM_TRIALS}"
            / name
            / split
            / f"{source}.npz"
        ),
        "error_thresholds": root / "thresholds" / f"n{NUM_TRIALS}" / f"{name}.npz",
        "diagnostic_thresholds": (
            root / "diagnostics" / f"n{NUM_TRIALS}" / name / "thresholds.npz"
        ),
        "gold_archive": root / "gold" / split / f"{source}.npz",
        "gold_directory": _gold_directory(
            split=split,
            source=source,
            result_dir=root,
        ),
    }


def load_npz(path, *, keys=None):
    """Load one NPZ archive as a plain dictionary."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}")

    with np.load(path, allow_pickle=False) as archive:
        return {
            key: archive[key]
            for key in archive.files
            if keys is None or key in keys
        }


def _required(values, keys, name):
    missing = set(keys) - set(values)
    if missing:
        raise ValueError(f"{name}: missing keys {sorted(missing)}")


def _bool_array(values, *, name):
    array = np.asarray(values)
    if array.dtype.kind == "b":
        return array.astype(bool, copy=False)

    text = np.char.lower(array.astype(str))
    valid = ("true", "false", "1", "0", "t", "f")
    if not np.isin(text, valid).all():
        raise ValueError(f"{name}: invalid boolean values")
    return np.isin(text, ("true", "1", "t"))


def _vector(values, key, n, *, default=None, dtype=None):
    value = values[key] if key in values else default
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = np.full(n, array.item(), dtype=dtype)
    if array.shape != (n,):
        raise ValueError(f"{key}: expected {(n,)}, got {array.shape}")
    return array


def _matrix(values, key, n):
    array = np.asarray(values[key], dtype=float)
    expected = (n, len(ASSUMED_MODELS))
    if array.shape != expected:
        raise ValueError(f"{key}: expected {expected}, got {array.shape}")
    if np.isinf(array).any():
        raise ValueError(f"{key}: infinite values are invalid")
    return array


def _threshold(values, key, *, shared=False):
    array = np.asarray(values[key], dtype=float)

    if shared:
        if array.size != 1:
            raise ValueError(f"{key}: direct threshold must contain one value")
        value = float(array.reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError(f"{key}: threshold must be finite")
        return np.full(len(ASSUMED_MODELS), value, dtype=float)

    expected = (len(ASSUMED_MODELS),)
    if array.shape != expected or not np.isfinite(array).all():
        raise ValueError(f"{key}: expected finite thresholds with shape {expected}")
    return array


def check_alignment(metrics, diagnostics, inference, gold, *, split):
    """Check dataset order and candidate-model order across saved products."""
    _required(metrics, ("id", "candidate_models"), "metrics")

    ids = np.asarray(metrics["id"]).astype(str)
    if (
        ids.ndim != 1
        or len(ids) == 0
        or len(np.unique(ids)) != len(ids)
        or np.isin(ids, ("", "nan")).any()
    ):
        raise ValueError("Metric IDs must be a nonempty, unique string vector")

    models = np.asarray(metrics["candidate_models"]).astype(str)
    if not np.array_equal(models, np.asarray(ASSUMED_MODELS)):
        raise ValueError(f"Candidate order must be {ASSUMED_MODELS}")

    for name, values in (
        ("diagnostics", diagnostics),
        ("inference", inference),
        ("gold", gold),
    ):
        _required(values, ("id",), name)
        other_ids = np.asarray(values["id"]).astype(str)
        if not np.array_equal(ids, other_ids):
            raise ValueError(f"Dataset IDs/order do not match: {name}")

        if name != "diagnostics" or "candidate_models" in values:
            _required(values, ("candidate_models",), name)
            other_models = np.asarray(values["candidate_models"]).astype(str)
            if not np.array_equal(models, other_models):
                raise ValueError(f"Candidate order does not match: {name}")

        # Empirical observations have no true generating model / well-specified label.
        if split != "empirical":
            for key in ("generating_model", "well_specified"):
                if key in metrics and key in values:
                    left = _vector(metrics, key, len(ids))
                    right = _vector(values, key, len(ids))
                    if key == "well_specified":
                        left = _bool_array(left, name=f"metrics.{key}")
                        right = _bool_array(right, name=f"{name}.{key}")
                    if not np.array_equal(left, right):
                        raise ValueError(f"{key} does not match: {name}")


def load_gold(paths, ids):
    """Load bridge-sampling gold PMP/logML and convergence metadata."""
    if paths["gold_archive"].is_file():
        gold = load_npz(paths["gold_archive"])
        _required(gold, ("id", "candidate_models", "pmp"), "gold archive")
        return gold

    ids = np.asarray(ids).astype(str)
    n = len(ids)
    num_models = len(ASSUMED_MODELS)

    logml = np.empty((n, num_models), dtype=float)
    bridge_sd = np.full((n, num_models), np.nan, dtype=float)
    converged = np.zeros((n, num_models), dtype=bool)
    available = np.zeros((n, num_models), dtype=bool)

    for j, model in enumerate(ASSUMED_MODELS):
        model_root = Path(paths["gold_directory"]) / model
        bridge_path = model_root / "bridgesampling.csv"
        if not bridge_path.is_file():
            raise FileNotFoundError(
                f"Gold archive or bridge table is required: {bridge_path}"
            )

        table = pd.read_csv(bridge_path, dtype={"id": str}, keep_default_na=False)
        _required(table, ("id", "estimate"), str(bridge_path))
        table = table.set_index("id", verify_integrity=True)

        if set(table.index) != set(ids):
            raise ValueError(f"Gold IDs do not match metrics: {bridge_path}")
        table = table.loc[ids]

        logml[:, j] = pd.to_numeric(table["estimate"], errors="raise").to_numpy()
        if "sd" in table.columns:
            bridge_sd[:, j] = pd.to_numeric(table["sd"], errors="raise").to_numpy()

        if "converged" in table.columns:
            converged[:, j] = _bool_array(
                table["converged"].to_numpy(),
                name=str(bridge_path),
            )
            available[:, j] = True
        else:
            convergence_path = model_root / "convergence_diagnostics.csv"
            if convergence_path.is_file():
                status = pd.read_csv(
                    convergence_path,
                    dtype={"id": str},
                    keep_default_na=False,
                ).set_index("id", verify_integrity=True)
                if set(status.index) != set(ids):
                    raise ValueError(
                        f"Convergence IDs do not match metrics: {convergence_path}"
                    )
                _required(status, ("converged",), str(convergence_path))
                converged[:, j] = _bool_array(
                    status.loc[ids, "converged"].to_numpy(),
                    name=str(convergence_path),
                )
                available[:, j] = True

    if not np.isfinite(logml).all():
        raise ValueError("Bridge logML estimates must be finite")

    return {
        "id": ids,
        "candidate_models": np.asarray(ASSUMED_MODELS),
        "logml": logml,
        "pmp": softmax(logml, axis=-1),
        "bridge_sd": bridge_sd,
        "converged": converged,
        "convergence_available": available,
    }


def _gold_validity(gold, metrics, n, policy):
    """Return convergence status, availability, and validity mask."""
    if policy not in ("available", "strict", "all"):
        raise ValueError("convergence_policy must be available, strict, or all")

    shape = (n, len(ASSUMED_MODELS))
    status = gold if "converged" in gold else metrics

    converged = _bool_array(
        status.get("converged", np.zeros(shape, dtype=bool)),
        name="converged",
    )
    available = _bool_array(
        status.get(
            "convergence_available",
            np.full(shape, "converged" in status, dtype=bool),
        ),
        name="convergence_available",
    )

    if converged.shape != shape or available.shape != shape:
        raise ValueError("Convergence metadata must have shape (datasets, candidates)")

    if policy == "all":
        valid = np.ones(shape, dtype=bool)
    elif policy == "strict":
        valid = converged & available
    else:
        # Unknown convergence is not treated as failure.
        valid = (converged & available) | (~available)

    return converged, available, valid


def _positive(error, lower, upper, valid):
    flag = (error < lower) | (error > upper)
    return pd.array(np.where(valid, flag, None), dtype="boolean")


def _sources(split, sources, paths):
    if sources is not None:
        selected = sources.get(split, ()) if isinstance(sources, dict) else sources
    elif split == "simulated":
        selected = SIMULATED_DATASETS
    elif split == "benchmark":
        selected = ASSUMED_MODELS
    else:
        selected = tuple(
            p.stem for p in sorted(Path(paths["metrics"]).parent.glob("*.npz"))
        ) or ("empirical",)

    if isinstance(selected, str):
        selected = (selected,)
    selected = tuple(selected)

    if not selected or len(set(selected)) != len(selected):
        raise ValueError(f"{split}: sources must be nonempty and unique")
    return selected


def _empirical_split_is_absent(*, options):
    probe = input_paths(split="empirical", source="empirical", **options)
    required = (
        probe["metrics"],
        probe["inference"],
        probe["diagnostics"],
        probe["gold_directory"],
    )
    return not any(Path(path).exists() for path in required)


def build_comparison(
    *,
    method,
    summary_multiplier=None,
    scoring_rule=None,
    splits=("simulated", "empirical"),
    sources=None,
    result_dir=RESULT_DIR,
    convergence_policy="available",
    allow_missing_empirical=False,
):
    """Build one row per dataset × candidate model.

    No inference error or diagnostic is recalculated. The function validates
    saved results, attaches calibrated thresholds, and creates consistent flags.
    """
    options = {
        "method": method,
        "summary_multiplier": summary_multiplier,
        "scoring_rule": scoring_rule,
        "result_dir": result_dir,
    }
    run_name(
        method=method,
        summary_multiplier=summary_multiplier,
        scoring_rule=scoring_rule,
    )

    if isinstance(splits, str):
        splits = (splits,)
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("splits must be nonempty and unique")

    if "empirical" in splits and allow_missing_empirical:
        if _empirical_split_is_absent(options=options):
            warnings.warn(
                "Empirical inputs are absent; empirical rows will be skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
            splits = tuple(split for split in splits if split != "empirical")

    if not splits:
        raise ValueError("No evaluation split remains after applying missing-data rules")

    # Run-level thresholds do not depend on evaluation split.
    probe = input_paths(split=splits[0], source="_", **options)
    error_thresholds = load_npz(probe["error_thresholds"])
    diagnostic_thresholds = load_npz(probe["diagnostic_thresholds"])

    for name, thresholds in (
        ("error", error_thresholds),
        ("diagnostic", diagnostic_thresholds),
    ):
        if "candidate_models" in thresholds:
            models = np.asarray(thresholds["candidate_models"]).astype(str)
            if not np.array_equal(models, np.asarray(ASSUMED_MODELS)):
                raise ValueError(f"{name} threshold candidate order does not match")

    bounds = {}
    metrics_to_attach = ("pmp",) if method == "direct" else tuple(ERROR_SPECS)
    for metric in metrics_to_attach:
        prefix = "posterior_mmd" if metric == "posterior_mmd" else f"{metric}_error"
        low_key = f"{prefix}_lower"
        high_key = f"{prefix}_upper"
        _required(error_thresholds, (low_key, high_key), "error thresholds")
        low = _threshold(error_thresholds, low_key)
        high = _threshold(error_thresholds, high_key)
        if np.any(low > high):
            raise ValueError(f"{metric}: lower threshold exceeds upper threshold")
        bounds[metric] = (low, high)

    diagnostic_bounds = {}
    for diagnostic in DIAGNOSTICS:
        low_key = f"d_{diagnostic}_lower"
        high_key = f"d_{diagnostic}_upper"
        _required(
            diagnostic_thresholds,
            (low_key, high_key),
            "diagnostic thresholds",
        )
        low = _threshold(
            diagnostic_thresholds,
            low_key,
            shared=(method == "direct"),
        )
        high = _threshold(
            diagnostic_thresholds,
            high_key,
            shared=(method == "direct"),
        )
        if np.any(high <= 0) or np.any(low > high):
            raise ValueError(
                f"{diagnostic}: upper threshold must be positive and lower <= upper"
            )
        diagnostic_bounds[diagnostic] = (low, high)

    frames = []

    for split in splits:
        split_probe = input_paths(split=split, source="_", **options)
        selected_sources = _sources(split, sources, split_probe)

        for source in selected_sources:
            paths = input_paths(split=split, source=source, **options)

            metrics = load_npz(paths["metrics"])
            _required(
                metrics,
                ("id", "candidate_models", "signed_pmp_error"),
                "metrics",
            )

            ids = np.asarray(metrics["id"]).astype(str)
            n = len(ids)

            diagnostics = load_npz(paths["diagnostics"])
            if method == "indirect":
                _required(diagnostics, ("candidate_models",), "indirect diagnostics")

            inference = load_npz(
                paths["inference"],
                keys=(
                    "id",
                    "candidate_models",
                    "generating_model",
                    "well_specified",
                    "pmp",
                    "logml",
                ),
            )

            gold = load_gold(paths, ids)
            check_alignment(
                metrics,
                diagnostics,
                inference,
                gold,
                split=split,
            )

            _required(inference, ("pmp",), "inference")
            estimated_pmp = _matrix(inference, "pmp", n)
            gold_pmp = _matrix(gold, "pmp", n)

            for name, matrix in (
                ("estimated", estimated_pmp),
                ("gold", gold_pmp),
            ):
                if (
                    not np.isfinite(matrix).all()
                    or np.any((matrix < 0) | (matrix > 1))
                    or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8)
                ):
                    raise ValueError(
                        f"{name} PMP must be finite probabilities summing to one"
                    )

            pmp_error = _matrix(metrics, "signed_pmp_error", n)
            if not np.allclose(
                pmp_error,
                estimated_pmp - gold_pmp,
                rtol=1e-6,
                atol=1e-7,
            ):
                raise ValueError(
                    f"Saved PMP errors disagree with inference/gold: {split}/{source}"
                )

            converged, available, gold_valid = _gold_validity(
                gold,
                metrics,
                n,
                convergence_policy,
            )

            pmp_valid = np.isfinite(pmp_error).all(axis=1) & gold_valid.all(axis=1)
            pmp_low, pmp_high = bounds["pmp"]
            any_pmp_error = (
                (pmp_error < pmp_low[None, :])
                | (pmp_error > pmp_high[None, :])
            ).any(axis=1)

            if split == "empirical":
                generating = np.full(n, "", dtype=str)
                well = np.zeros(n, dtype=bool)
                well_nullable = pd.array([None] * n, dtype="boolean")
            else:
                _required(
                    metrics,
                    ("generating_model", "well_specified"),
                    f"{split} metrics",
                )
                generating = _vector(
                    metrics,
                    "generating_model",
                    n,
                    dtype=str,
                )
                well = _bool_array(
                    _vector(metrics, "well_specified", n),
                    name="well_specified",
                )
                well_nullable = pd.array(well, dtype="boolean")

            metric_values = {"pmp": pmp_error}

            if method == "indirect":
                _required(
                    metrics,
                    ("signed_logml_error", "posterior_mmd"),
                    "indirect metrics",
                )
                _required(inference, ("logml",), "indirect inference")
                _required(gold, ("logml",), "indirect gold")

                logml_error = _matrix(metrics, "signed_logml_error", n)
                posterior_mmd = _matrix(metrics, "posterior_mmd", n)

                expected_logml_error = (
                    _matrix(inference, "logml", n) - _matrix(gold, "logml", n)
                )
                if not np.allclose(
                    logml_error,
                    expected_logml_error,
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"Saved logML errors disagree with inference/gold: {split}/{source}"
                    )

                if np.any(posterior_mmd[np.isfinite(posterior_mmd)] < 0):
                    raise ValueError("posterior_mmd must be nonnegative")

                metric_values.update(
                    logml=logml_error,
                    posterior_mmd=posterior_mmd,
                )

            diagnostic_values = {}
            for diagnostic in DIAGNOSTICS:
                score_key = f"d_{diagnostic}"
                rho_key = f"rho_{diagnostic}"
                typical_key = f"typical_{diagnostic}"

                _required(diagnostics, (score_key,), "diagnostics")
                score = np.asarray(diagnostics[score_key], dtype=float)
                expected_shape = (
                    (n,)
                    if method == "direct"
                    else (n, len(ASSUMED_MODELS))
                )
                if score.shape != expected_shape or not np.isfinite(score).all():
                    raise ValueError(
                        f"{score_key}: expected finite values with shape {expected_shape}"
                    )

                if method == "direct":
                    score = np.repeat(score[:, None], len(ASSUMED_MODELS), axis=1)

                low, high = diagnostic_bounds[diagnostic]
                rho = score / high[None, :]
                typical = (
                    (score >= low[None, :])
                    & (score <= high[None, :])
                )

                if rho_key in diagnostics:
                    cached = np.asarray(diagnostics[rho_key], dtype=float)
                    if method == "direct":
                        if cached.shape != (n,):
                            raise ValueError(f"{rho_key}: expected shape {(n,)}")
                        cached = np.repeat(
                            cached[:, None],
                            len(ASSUMED_MODELS),
                            axis=1,
                        )
                    if cached.shape != rho.shape or not np.allclose(
                        cached,
                        rho,
                        rtol=1e-5,
                        atol=1e-7,
                    ):
                        raise ValueError(
                            f"{rho_key} disagrees with d / upper threshold"
                        )

                if typical_key in diagnostics:
                    cached_typical = _bool_array(
                        diagnostics[typical_key],
                        name=typical_key,
                    )
                    if method == "direct":
                        if cached_typical.shape != (n,):
                            raise ValueError(
                                f"{typical_key}: expected shape {(n,)}"
                            )
                        cached_typical = np.repeat(
                            cached_typical[:, None],
                            len(ASSUMED_MODELS),
                            axis=1,
                        )
                    if cached_typical.shape != typical.shape or not np.array_equal(
                        cached_typical,
                        typical,
                    ):
                        raise ValueError(
                            f"{typical_key} disagrees with calibrated interval"
                        )

                diagnostic_values[diagnostic] = (score, rho, typical)

            for j, model in enumerate(ASSUMED_MODELS):
                model_matched = (
                    (generating == model) & well
                    if split != "empirical"
                    else np.zeros(n, dtype=bool)
                )

                frame = pd.DataFrame(
                    {
                        "method": method,
                        "num_obs": NUM_TRIALS,
                        "summary_multiplier": (
                            summary_multiplier if method == "indirect" else 0
                        ),
                        "summary_dim": (
                            PARAM_DIMS[model] * summary_multiplier
                            if method == "indirect"
                            else DIRECT_SUMMARY_DIM
                        ),
                        "scoring_rule": scoring_rule or "",
                        "split": split,
                        "dataset_type": split,
                        "source_model": source,
                        "generating_model": generating,
                        "id": ids,
                        "candidate_model": model,
                        "well_specified": well_nullable,
                        "model_matched": model_matched,
                        "estimated_pmp": estimated_pmp[:, j],
                        "gold_pmp": gold_pmp[:, j],
                        "absolute_pmp_error": np.abs(pmp_error[:, j]),
                        "pmp_l1_error": np.abs(pmp_error).sum(axis=1),
                        "high_pmp_error_any": pd.array(
                            np.where(pmp_valid, any_pmp_error, None),
                            dtype="boolean",
                        ),
                        "gold_converged": pd.array(
                            np.where(available[:, j], converged[:, j], None),
                            dtype="boolean",
                        ),
                        "convergence_available": available[:, j],
                        "all_convergence_available": available.all(axis=1),
                        "all_gold_converged": pd.array(
                            np.where(
                                available.all(axis=1),
                                converged.all(axis=1),
                                None,
                            ),
                            dtype="boolean",
                        ),
                        "convergence_policy": convergence_policy,
                    }
                )

                for metric, matrix in metric_values.items():
                    value_col, low_col, high_col = ERROR_SPECS[metric]
                    low, high = bounds[metric]
                    error = matrix[:, j]

                    valid = (
                        pmp_valid
                        if metric == "pmp"
                        else np.isfinite(error) & gold_valid[:, j]
                    )
                    prefix = "posterior" if metric == "posterior_mmd" else metric

                    frame[value_col] = error
                    frame[low_col] = low[j]
                    frame[high_col] = high[j]
                    frame[f"{prefix}_error_valid"] = valid
                    frame[f"{prefix}_error_positive"] = _positive(
                        error,
                        low[j],
                        high[j],
                        valid,
                    )

                if method == "indirect":
                    frame["estimated_logml"] = _matrix(inference, "logml", n)[:, j]
                    frame["gold_logml"] = _matrix(gold, "logml", n)[:, j]
                    frame["absolute_logml_error"] = np.abs(
                        metric_values["logml"][:, j]
                    )
                    frame["posterior_mmd_available"] = np.isfinite(
                        metric_values["posterior_mmd"][:, j]
                    )

                    for column in ("importance_ess", "importance_ess_ratio"):
                        if column in metrics:
                            frame[column] = _matrix(metrics, column, n)[:, j]

                for diagnostic, (score, rho, typical) in diagnostic_values.items():
                    low, high = diagnostic_bounds[diagnostic]
                    frame[f"d_{diagnostic}"] = score[:, j]
                    frame[f"rho_{diagnostic}"] = rho[:, j]
                    frame[f"{diagnostic}_lower"] = low[j]
                    frame[f"{diagnostic}_upper"] = high[j]
                    frame[f"rho_low_{diagnostic}"] = low[j] / high[j]
                    frame[f"typical_{diagnostic}"] = typical[:, j]
                    frame[f"high_surprise_{diagnostic}"] = rho[:, j] > 1.0
                    frame[f"all_high_surprise_{diagnostic}"] = (
                        rho > 1.0
                    ).all(axis=1)
                    frame[f"at_least_one_not_high_surprise_{diagnostic}"] = (
                        rho <= 1.0
                    ).any(axis=1)

                frames.append(frame)

    if not frames:
        raise ValueError("No evaluation inputs were selected")

    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(KEYS).any():
        duplicate = data.loc[data.duplicated(KEYS, keep=False), KEYS].head()
        raise ValueError(
            "Duplicate dataset/candidate keys across splits:\n"
            + duplicate.to_string(index=False)
        )

    return data


def build_indirect_comparison(*, summary_multiplier, **kwargs):
    return build_comparison(
        method="indirect",
        summary_multiplier=summary_multiplier,
        **kwargs,
    )


def build_direct_comparison(*, scoring_rule, **kwargs):
    return build_comparison(
        method="direct",
        scoring_rule=scoring_rule,
        **kwargs,
    )


def binary_metrics(actual, predicted, score):
    """Compute confusion counts, FNR/FPR, and rank-based ROC AUC."""
    actual = pd.Series(actual, dtype="boolean").reset_index(drop=True)
    predicted = np.asarray(predicted, dtype=bool)
    score = np.asarray(score, dtype=float)

    if predicted.shape != score.shape or predicted.shape != (len(actual),):
        raise ValueError("Actual labels, predictions, and scores must align")

    keep = actual.notna().to_numpy() & np.isfinite(score)
    truth = actual.loc[keep].to_numpy(bool)
    pred = predicted[keep]
    values = score[keep]

    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    tn = int(np.sum(~truth & ~pred))

    positive = int(truth.sum())
    negative = int((~truth).sum())

    auc = np.nan
    if positive > 0 and negative > 0:
        ranks = rankdata(values, method="average")
        auc = float(
            (ranks[truth].sum() - positive * (positive + 1) / 2)
            / (positive * negative)
        )

    return {
        "n_total": len(actual),
        "n": int(keep.sum()),
        "n_excluded": int((~keep).sum()),
        "n_error_positive": positive,
        "n_error_negative": negative,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "FNR": fn / positive if positive else np.nan,
        "FPR": fp / negative if negative else np.nan,
        "AUC": auc,
    }


def _data_groups(data):
    yield "all", data
    for name, group in data.groupby("dataset_type", sort=False, dropna=False):
        yield str(name), group


def component_detection(data, *, method=None):
    """Candidate-level diagnostic detection for each error metric."""
    method = method or str(data["method"].iloc[0])
    rows = []

    metrics = ("pmp",) if method == "direct" else tuple(ERROR_SPECS)

    for dataset_type, group in _data_groups(data):
        for model, panel in group.groupby("candidate_model", sort=False):
            for metric in metrics:
                _, low_col, high_col = ERROR_SPECS[metric]
                prefix = "posterior" if metric == "posterior_mmd" else metric

                for diagnostic in DIAGNOSTICS:
                    rho = panel[f"rho_{diagnostic}"].to_numpy(float)
                    rows.append(
                        {
                            "method": method,
                            "dataset_type": dataset_type,
                            "scope": "component",
                            "candidate_model": model,
                            "error_metric": metric,
                            "diagnostic": diagnostic,
                            "error_lower": panel[low_col].iloc[0],
                            "error_upper": panel[high_col].iloc[0],
                            "rho_threshold": 1.0,
                            **binary_metrics(
                                panel[f"{prefix}_error_positive"],
                                rho > 1.0,
                                rho,
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def global_pmp_detection(data, *, method=None):
    """Dataset-level diagnostic detection of any high PMP component error.

    For indirect inference the primary four-model rule is ``all_high_surprise``:
    min_j rho_j > 1. ``any_high_surprise`` is retained as a sensitivity rule.
    Direct diagnostics are shared across candidates, so only one rule is needed.
    """
    method = method or str(data["method"].iloc[0])
    rows = []

    rules = (
        (("all_high_surprise", "min"), ("any_high_surprise", "max"))
        if method == "indirect"
        else (("shared", "max"),)
    )

    for dataset_type, group in _data_groups(data):
        grouped = group.groupby(DATASET_KEYS, sort=False, dropna=False)
        if not grouped.size().eq(len(ASSUMED_MODELS)).all():
            raise ValueError(
                "Global PMP detection requires all four candidates for every dataset"
            )

        actual = grouped["high_pmp_error_any"].agg(lambda x: x.iloc[0])

        for diagnostic in DIAGNOSTICS:
            for rule, aggregate in rules:
                rho = grouped[f"rho_{diagnostic}"].agg(aggregate).to_numpy(float)
                rows.append(
                    {
                        "method": method,
                        "dataset_type": dataset_type,
                        "scope": "dataset",
                        "error_metric": "pmp_any_component",
                        "diagnostic": diagnostic,
                        "diagnostic_rule": rule,
                        "rho_threshold": 1.0,
                        **binary_metrics(
                            actual,
                            rho > 1.0,
                            rho,
                        ),
                    }
                )

    return pd.DataFrame(rows)


def run_comparison(
    *,
    method,
    summary_multiplier=None,
    scoring_rule=None,
    overwrite=False,
    **kwargs,
):
    paths = comparison_paths(
        method=method,
        summary_multiplier=summary_multiplier,
        scoring_rule=scoring_rule,
        result_dir=kwargs.get("result_dir", RESULT_DIR),
    )

    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Comparison already exists: {existing[0]}; use overwrite=True to rebuild"
        )

    data = build_comparison(
        method=method,
        summary_multiplier=summary_multiplier,
        scoring_rule=scoring_rule,
        **kwargs,
    )

    tables = {
        "data": data,
        "detection": component_detection(data, method=method),
        "global_pmp_detection": global_pmp_detection(data, method=method),
    }

    for name, frame in tables.items():
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(paths[name], index=False)

    print(f"Saved {method} comparison: {paths['data'].parent}")
    return tables


def run_indirect_comparison(*, summary_multiplier, **kwargs):
    return run_comparison(
        method="indirect",
        summary_multiplier=summary_multiplier,
        **kwargs,
    )


def run_direct_comparison(*, scoring_rule, **kwargs):
    return run_comparison(
        method="direct",
        scoring_rule=scoring_rule,
        **kwargs,
    )


def run_all_comparisons(
    *,
    summary_multipliers=(1, 2, 4),
    scoring_rules=SCORING_RULES,
    **kwargs,
):
    outputs = {}

    for multiplier in summary_multipliers:
        outputs[f"indirect_s{multiplier}D"] = run_indirect_comparison(
            summary_multiplier=multiplier,
            **kwargs,
        )

    for rule in scoring_rules:
        outputs[f"direct_{rule}_s{DIRECT_SUMMARY_DIM}"] = run_direct_comparison(
            scoring_rule=rule,
            **kwargs,
        )

    return outputs


def _parse_nullable_boolean(series):
    text = series.astype(str).str.lower()
    mapped = text.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "t": True,
            "f": False,
            "": None,
            "nan": None,
            "<na>": None,
        }
    )
    if mapped.isna().sum() != text.isin(("", "nan", "<na>")).sum():
        bad = sorted(set(text[mapped.isna()]) - {"", "nan", "<na>"})
        if bad:
            raise ValueError(f"Invalid boolean values in CSV: {bad}")
    return pd.array(mapped, dtype="boolean")


def load_comparison(
    *,
    method,
    summary_multiplier=None,
    scoring_rule=None,
    result_dir=RESULT_DIR,
):
    """Load one saved comparison run with stable dtypes."""
    paths = comparison_paths(
        method=method,
        summary_multiplier=summary_multiplier,
        scoring_rule=scoring_rule,
        result_dir=result_dir,
    )

    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing comparison file: {missing[0]}")

    tables = {
        name: pd.read_csv(
            path,
            dtype={"id": str, "source_model": str},
            keep_default_na=False,
        )
        for name, path in paths.items()
    }

    data = tables["data"]
    boolean_columns = []
    for column in data.columns:
        if (
            column.startswith(
                (
                    "typical_",
                    "high_surprise_",
                    "all_high_surprise_",
                    "at_least_one_not_high_surprise_",
                )
            )
            or column.endswith(("_valid", "_positive", "_available"))
            or column
            in (
                "well_specified",
                "model_matched",
                "high_pmp_error_any",
                "gold_converged",
                "all_gold_converged",
            )
        ):
            boolean_columns.append(column)

    for column in boolean_columns:
        data[column] = _parse_nullable_boolean(data[column])

    string_columns = {
        "method",
        "scoring_rule",
        "split",
        *KEYS,
        "generating_model",
        "convergence_policy",
    }
    for column in data.columns:
        if column not in string_columns and column not in boolean_columns:
            series = data[column].mask(data[column].eq(""), np.nan)
            data[column] = pd.to_numeric(series, errors="raise")

    categorical = {
        "method",
        "dataset_type",
        "scope",
        "candidate_model",
        "error_metric",
        "diagnostic",
        "diagnostic_rule",
    }
    for name in ("detection", "global_pmp_detection"):
        table = tables[name]
        for column in table.columns:
            if column not in categorical:
                series = table[column].mask(table[column].eq(""), np.nan)
                table[column] = pd.to_numeric(series, errors="raise")

    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("indirect", "direct", "both"),
        default="both",
    )
    parser.add_argument(
        "--summary-multipliers",
        nargs="+",
        type=int,
        default=[1, 2, 4],
    )
    parser.add_argument(
        "--scoring-rules",
        nargs="+",
        choices=SCORING_RULES,
        default=list(SCORING_RULES),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("simulated", "empirical", "benchmark"),
        default=["simulated", "empirical"],
    )
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument(
        "--convergence-policy",
        choices=("available", "strict", "all"),
        default="available",
    )
    parser.add_argument("--allow-missing-empirical", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_all_comparisons(
        summary_multipliers=(
            args.summary_multipliers if args.method != "direct" else ()
        ),
        scoring_rules=(args.scoring_rules if args.method != "indirect" else ()),
        splits=tuple(args.splits),
        result_dir=args.result_dir,
        convergence_policy=args.convergence_policy,
        allow_missing_empirical=args.allow_missing_empirical,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

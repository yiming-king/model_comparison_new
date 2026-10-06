"""Join saved RDM diagnostics and inference errors; no neural-network imports.

Existing metric/inference paths follow the repository. Diagnostic paths follow
the Gaussian convention, with RDM run names such as ``indirect_s4D``.
See ``input_paths`` and the accompanying notebook for the exact input contract.
"""

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata

from ..config import ASSUMED_MODELS, NUM_TRIALS, PARAM_DIMS, RESULT_DIR, SIMULATED_DATASETS

DIAGNOSTICS = ("l2", "linf", "density", "mmd")
DIRECT_SUMMARY_DIM = 12
SCORING_RULES = ("cross_entropy", "logistic", "exponential")
KEYS = ["dataset_type", "source_model", "id", "candidate_model"]
DATASET_KEYS = KEYS[:-1]
ERROR_SPECS = {
    "posterior_mmd": ("posterior_mmd", "posterior_error_lower", "posterior_error_upper"),
    "logml": ("signed_logml_error", "logml_error_lower", "logml_error_upper"),
    "pmp": ("signed_pmp_error", "pmp_error_lower", "pmp_error_upper"),
}


def run_name(*, method, summary_multiplier=None, scoring_rule=None):
    if method == "indirect":
        if summary_multiplier not in (1, 2, 4):
            raise ValueError("summary_multiplier must be 1, 2, or 4")
        return f"indirect_s{summary_multiplier}D"
    if method == "direct" and scoring_rule in SCORING_RULES:
        return f"direct_{scoring_rule}_s{DIRECT_SUMMARY_DIM}"
    raise ValueError("Use method='indirect' or method='direct' with a valid scoring_rule")


def comparison_paths(*, method, summary_multiplier=None, scoring_rule=None, result_dir=RESULT_DIR):
    name = run_name(method=method, summary_multiplier=summary_multiplier, scoring_rule=scoring_rule)
    root = Path(result_dir) / "comparison" / f"n{NUM_TRIALS}" / name
    return {key: root / f"{key}.csv" for key in ("data", "detection", "global_pmp_detection")}


def input_paths(*, method, split, source, summary_multiplier=None, scoring_rule=None, result_dir=RESULT_DIR):
    """Gold accepts an NPZ archive or the repository's per-model bridge CSVs.

    Metrics/inference: PRODUCT/n384/SPLIT/RUN/SOURCE.npz.
    Diagnostics: diagnostics/n384/RUN/SPLIT/SOURCE.npz.
    Error thresholds: thresholds/n384/RUN.npz.
    Diagnostic thresholds: diagnostics/n384/RUN/thresholds.npz.
    """
    if split not in ("simulated", "empirical", "benchmark"):
        raise ValueError("split must be simulated, empirical, or benchmark")
    name = run_name(method=method, summary_multiplier=summary_multiplier, scoring_rule=scoring_rule)
    root = Path(result_dir)
    relative = Path(f"n{NUM_TRIALS}") / split / name / f"{source}.npz"
    return {
        "metrics": root / "metrics" / relative,
        "inference": root / "inference" / relative,
        "diagnostics": root / "diagnostics" / f"n{NUM_TRIALS}" / name / split / f"{source}.npz",
        "error_thresholds": root / "thresholds" / f"n{NUM_TRIALS}" / f"{name}.npz",
        "diagnostic_thresholds": root / "diagnostics" / f"n{NUM_TRIALS}" / name / "thresholds.npz",
        "gold_archive": root / "gold" / split / f"{source}.npz",
        "gold_directory": root / "gold" / split / source,
    }


def load_npz(path, *, keys=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files if keys is None or key in keys}


def _required(values, keys, name):
    missing = set(keys) - set(values)
    if missing:
        raise ValueError(f"{name}: missing keys {sorted(missing)}")


def _bool_array(values, *, name):
    array = np.asarray(values)
    if array.dtype.kind == "b":
        return array
    text = np.char.lower(array.astype(str))
    if not np.isin(text, ("true", "false", "1", "0", "t", "f")).all():
        raise ValueError(f"{name}: invalid boolean values")
    return np.isin(text, ("true", "1", "t"))


def _vector(values, key, n, *, default=None, dtype=None):
    array = np.asarray(values.get(key, default), dtype=dtype)
    if array.ndim == 0:
        array = np.full(n, array.item(), dtype=dtype)
    if array.shape != (n,):
        raise ValueError(f"{key}: expected {(n,)}, got {array.shape}")
    return array


def _matrix(values, key, n):
    array = np.asarray(values[key], dtype=float)
    if array.shape != (n, len(ASSUMED_MODELS)):
        raise ValueError(f"{key}: expected {(n, len(ASSUMED_MODELS))}, got {array.shape}")
    if np.isinf(array).any():
        raise ValueError(f"{key}: infinite values are invalid")
    return array


def _threshold(values, key, *, shared=False):
    array = np.asarray(values[key], dtype=float)
    expected = () if shared else (len(ASSUMED_MODELS),)
    if shared and array.size == 1:
        array = array.reshape(())
    if array.shape != expected or not np.isfinite(array).all():
        raise ValueError(f"{key}: expected finite thresholds with shape {expected}")
    return np.full(len(ASSUMED_MODELS), array.item()) if shared else array


def check_alignment(metrics, diagnostics, inference, gold):
    """Reject duplicate IDs, reordered rows, and reordered candidate columns."""
    _required(metrics, ("id", "candidate_models"), "metrics")
    ids = np.asarray(metrics["id"]).astype(str)
    if ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids) or np.isin(ids, ("", "nan")).any():
        raise ValueError("Metric IDs must be a nonempty, unique string vector")
    models = np.asarray(metrics["candidate_models"]).astype(str)
    if not np.array_equal(models, ASSUMED_MODELS):
        raise ValueError(f"Candidate order must be {ASSUMED_MODELS}")
    for name, values in (("diagnostics", diagnostics), ("inference", inference), ("gold", gold)):
        _required(values, ("id",), name)
        if not np.array_equal(ids, np.asarray(values["id"]).astype(str)):
            raise ValueError(f"Dataset IDs/order do not match: {name}")
        if name != "diagnostics" or "candidate_models" in values:
            _required(values, ("candidate_models",), name)
            if not np.array_equal(models, np.asarray(values["candidate_models"]).astype(str)):
                raise ValueError(f"Candidate order does not match: {name}")
        for key in ("generating_model", "well_specified"):
            if key in metrics and key in values:
                left = _vector(metrics, key, len(ids))
                right = _vector(values, key, len(ids))
                if key == "well_specified":
                    left, right = _bool_array(left, name=key), _bool_array(right, name=key)
                if not np.array_equal(left, right):
                    raise ValueError(f"{key} does not match: {name}")


def load_gold(paths, ids):
    """Read current gold without loading observations, Stan draws, or networks."""
    if paths["gold_archive"].is_file():
        gold = load_npz(paths["gold_archive"])
        _required(gold, ("id", "candidate_models", "pmp"), "gold archive")
        return gold
    n = len(ids)
    logml = np.empty((n, len(ASSUMED_MODELS)))
    bridge_sd = np.full_like(logml, np.nan)
    converged = np.zeros_like(logml, dtype=bool)
    available = np.zeros_like(converged)
    for j, model in enumerate(ASSUMED_MODELS):
        root = paths["gold_directory"] / model
        path = root / "bridgesampling.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Gold archive or bridge table is required: {path}")
        table = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
        _required(table, ("id", "estimate"), str(path))
        table = table.set_index("id", verify_integrity=True)
        if set(table.index) != set(ids):
            raise ValueError(f"Gold IDs do not match metrics: {path}")
        table = table.loc[ids]
        logml[:, j] = table["estimate"].to_numpy(float)
        if "sd" in table:
            bridge_sd[:, j] = pd.to_numeric(table["sd"], errors="raise")
        if "converged" in table:
            converged[:, j] = _bool_array(table["converged"], name=str(path))
            available[:, j] = True
        elif (root / "convergence_diagnostics.csv").is_file():
            status = pd.read_csv(root / "convergence_diagnostics.csv", dtype={"id": str})
            status = status.set_index("id", verify_integrity=True)
            if set(status.index) != set(ids):
                raise ValueError(f"Convergence IDs do not match metrics: {root}")
            converged[:, j] = _bool_array(status.loc[ids, "converged"], name=str(root))
            available[:, j] = True
    if not np.isfinite(logml).all():
        raise ValueError("Bridge logML estimates must be finite")
    return {
        "id": np.asarray(ids), "candidate_models": np.asarray(ASSUMED_MODELS),
        "logml": logml, "pmp": softmax(logml, axis=-1), "bridge_sd": bridge_sd,
        "converged": converged, "convergence_available": available,
    }


def _gold_validity(gold, metrics, n, policy):
    if policy not in ("available", "strict", "all"):
        raise ValueError("convergence_policy must be available, strict, or all")
    shape = (n, len(ASSUMED_MODELS))
    status = gold if "converged" in gold else metrics
    converged = _bool_array(status.get("converged", np.zeros(shape, dtype=bool)), name="converged")
    available = _bool_array(status.get("convergence_available", np.full(shape, "converged" in status, dtype=bool)),
                            name="convergence_available")
    if converged.shape != shape or available.shape != shape:
        raise ValueError("Convergence metadata must have shape (datasets, candidates)")
    valid = np.ones(shape, dtype=bool) if policy == "all" else converged & available
    if policy == "available":
        valid |= ~available  # Unknown is recorded separately, not called a failed fit.
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
        selected = tuple(p.stem for p in sorted(paths["metrics"].parent.glob("*.npz"))) or ("empirical",)
    if isinstance(selected, str):
        selected = (selected,)
    selected = tuple(selected)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(f"{split}: sources must be nonempty and unique")
    return selected


def build_comparison(*, method, summary_multiplier=None, scoring_rule=None,
                     splits=("simulated", "empirical"), sources=None, result_dir=RESULT_DIR,
                     convergence_policy="available", allow_missing_empirical=False):
    """Build one row per dataset/candidate. Saved logML stays in natural logs.

    No metric, threshold, or diagnostic is recalculated. Missing posterior MMD
    remains missing. Empirical generating-model labels are never fabricated.
    Missing empirical inputs can be explicitly allowed only when that entire
    split is absent; a partly completed split always raises an error.
    """
    options = dict(method=method, summary_multiplier=summary_multiplier,
                   scoring_rule=scoring_rule, result_dir=result_dir)
    run_name(**{k: options[k] for k in ("method", "summary_multiplier", "scoring_rule")})
    if isinstance(splits, str):
        splits = (splits,)
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("splits must be nonempty and unique")
    probe = input_paths(split=splits[0], source="_", **options)
    error_thresholds = load_npz(probe["error_thresholds"])
    diagnostic_thresholds = load_npz(probe["diagnostic_thresholds"])
    for name, thresholds in (("error", error_thresholds), ("diagnostic", diagnostic_thresholds)):
        if "candidate_models" in thresholds and not np.array_equal(thresholds["candidate_models"], ASSUMED_MODELS):
            raise ValueError(f"{name} threshold candidate order does not match")
    bounds = {}
    for metric in (("pmp",) if method == "direct" else ERROR_SPECS):
        prefix = "posterior_mmd" if metric == "posterior_mmd" else f"{metric}_error"
        _required(error_thresholds, (f"{prefix}_lower", f"{prefix}_upper"), "error thresholds")
        low, high = (_threshold(error_thresholds, f"{prefix}_{side}") for side in ("lower", "upper"))
        if np.any(low > high):
            raise ValueError(f"{metric}: lower threshold exceeds upper threshold")
        bounds[metric] = (low, high)
    diag_bounds = {}
    for diagnostic in DIAGNOSTICS:
        _required(diagnostic_thresholds, (f"d_{diagnostic}_lower", f"d_{diagnostic}_upper"), "diagnostic thresholds")
        low, high = (_threshold(diagnostic_thresholds, f"d_{diagnostic}_{side}", shared=method == "direct")
                     for side in ("lower", "upper"))
        if np.any(high <= 0) or np.any(low > high):
            raise ValueError(f"{diagnostic}: rho requires a positive upper threshold and lower <= upper")
        diag_bounds[diagnostic] = (low, high)

    frames = []
    for split in splits:
        paths = input_paths(split=split, source="_", **options)
        selected = _sources(split, sources, paths)
        inputs = [input_paths(split=split, source=source, **options) for source in selected]
        if split == "empirical" and allow_missing_empirical:
            roots = [paths[key].parent for key in ("metrics", "inference", "diagnostics")]
            gold_root = Path(result_dir) / "gold" / split
            absent = not any(root.exists() and any(root.iterdir()) for root in (*roots, gold_root))
            if absent:
                warnings.warn("Empirical inputs are absent; this comparison contains simulated data only.",
                              RuntimeWarning, stacklevel=2)
                continue
        for source, paths in zip(selected, inputs, strict=True):
            metrics = load_npz(paths["metrics"])
            _required(metrics, ("id", "candidate_models", "signed_pmp_error"), "metrics")
            ids = np.asarray(metrics["id"]).astype(str)
            n = len(ids)
            diagnostics = load_npz(paths["diagnostics"])
            if method == "indirect":
                _required(diagnostics, ("candidate_models",), "indirect diagnostics")
            inference = load_npz(paths["inference"], keys=("id", "candidate_models", "generating_model",
                                 "well_specified", "pmp", "logml"))
            gold = load_gold(paths, ids)
            check_alignment(metrics, diagnostics, inference, gold)
            _required(inference, ("pmp",), "inference")
            estimated_pmp, gold_pmp = _matrix(inference, "pmp", n), _matrix(gold, "pmp", n)
            for name, values in (("estimated", estimated_pmp), ("gold", gold_pmp)):
                if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)) or not np.allclose(values.sum(1), 1):
                    raise ValueError(f"{name} PMP must be finite probabilities summing to one")
            pmp_error = _matrix(metrics, "signed_pmp_error", n)
            if not np.allclose(pmp_error, estimated_pmp - gold_pmp, rtol=1e-6, atol=1e-7):
                raise ValueError(f"Saved PMP errors disagree with inference/gold: {split}/{source}")
            converged, available, gold_valid = _gold_validity(gold, metrics, n, convergence_policy)
            pmp_valid = np.isfinite(pmp_error).all(1) & gold_valid.all(1)
            pmp_low, pmp_high = bounds["pmp"]
            any_error = ((pmp_error < pmp_low) | (pmp_error > pmp_high)).any(1)
            if split != "empirical":
                _required(metrics, ("generating_model", "well_specified"), "simulated/benchmark metrics")
            generating = _vector(metrics, "generating_model", n, default="", dtype=str)
            well = _bool_array(_vector(metrics, "well_specified", n, default=False), name="well_specified")
            values = {"pmp": pmp_error}
            if method == "indirect":
                _required(metrics, ("signed_logml_error", "posterior_mmd"), "indirect metrics")
                _required(inference, ("logml",), "indirect inference")
                _required(gold, ("logml",), "indirect gold")
                values.update(logml=_matrix(metrics, "signed_logml_error", n),
                              posterior_mmd=_matrix(metrics, "posterior_mmd", n))
                if not np.allclose(values["logml"], _matrix(inference, "logml", n) - _matrix(gold, "logml", n),
                                   rtol=1e-6, atol=1e-6, equal_nan=True):
                    raise ValueError(f"Saved logML errors disagree with inference/gold: {split}/{source}")
                if np.any(values["posterior_mmd"] < 0):
                    raise ValueError("posterior_mmd must contain MMD, which is nonnegative")
            diag_values = {}
            for diagnostic in DIAGNOSTICS:
                _required(diagnostics, (f"d_{diagnostic}",), "diagnostics")
                score = np.asarray(diagnostics[f"d_{diagnostic}"], dtype=float)
                expected = (n,) if method == "direct" else (n, len(ASSUMED_MODELS))
                if score.shape != expected or not np.isfinite(score).all():
                    raise ValueError(f"d_{diagnostic}: expected finite scores of shape {expected}")
                if method == "direct":
                    score = np.repeat(score[:, None], len(ASSUMED_MODELS), axis=1)
                low, high = diag_bounds[diagnostic]
                rho = score / high
                if f"rho_{diagnostic}" in diagnostics:
                    cached = np.asarray(diagnostics[f"rho_{diagnostic}"], dtype=float)
                    if method == "direct":
                        if cached.shape != (n,):
                            raise ValueError(f"rho_{diagnostic}: expected shape {(n,)}")
                        cached = np.repeat(cached[:, None], len(ASSUMED_MODELS), axis=1)
                    if cached.shape != rho.shape or not np.allclose(cached, rho, rtol=1e-5, atol=1e-7):
                        raise ValueError(f"rho_{diagnostic} disagrees with d/upper threshold")
                diag_values[diagnostic] = (score, rho)
            for j, model in enumerate(ASSUMED_MODELS):
                frame = pd.DataFrame({
                    "method": method, "num_obs": NUM_TRIALS,
                    "summary_multiplier": summary_multiplier if method == "indirect" else 0,
                    "summary_dim": PARAM_DIMS[model] * summary_multiplier if method == "indirect" else DIRECT_SUMMARY_DIM,
                    "scoring_rule": scoring_rule or "", "split": split,
                    "dataset_type": "empirical" if split == "empirical" else "simulated",
                    "source_model": source, "generating_model": generating, "id": ids,
                    "candidate_model": model,
                    "well_specified": pd.array(well if split != "empirical" else [None] * n, dtype="boolean"),
                    "model_matched": (generating == model) & well & (split != "empirical"),
                    "estimated_pmp": estimated_pmp[:, j], "gold_pmp": gold_pmp[:, j],
                    "absolute_pmp_error": np.abs(pmp_error[:, j]), "pmp_l1_error": np.abs(pmp_error).sum(1),
                    "high_pmp_error_any": pd.array(np.where(pmp_valid, any_error, None), dtype="boolean"),
                    "gold_converged": pd.array(np.where(available[:, j], converged[:, j], None), dtype="boolean"),
                    "convergence_available": available[:, j], "all_convergence_available": available.all(1),
                    "all_gold_converged": pd.array(np.where(available.all(1), converged.all(1), None), dtype="boolean"),
                    "convergence_policy": convergence_policy,
                })
                for metric, matrix in values.items():
                    column, lowcol, highcol = ERROR_SPECS[metric]
                    low, high = bounds[metric]
                    error = matrix[:, j]
                    valid = pmp_valid if metric == "pmp" else np.isfinite(error) & gold_valid[:, j]
                    prefix = "posterior" if metric == "posterior_mmd" else metric
                    frame[column], frame[lowcol], frame[highcol] = error, low[j], high[j]
                    frame[f"{prefix}_error_valid"] = valid
                    frame[f"{prefix}_error_positive"] = _positive(error, low[j], high[j], valid)
                if method == "indirect":
                    frame["estimated_logml"] = inference["logml"][:, j]
                    frame["gold_logml"] = gold["logml"][:, j]
                    frame["absolute_logml_error"] = np.abs(values["logml"][:, j])
                    frame["posterior_mmd_available"] = np.isfinite(values["posterior_mmd"][:, j])
                    for column in ("importance_ess", "importance_ess_ratio"):
                        if column in metrics:
                            frame[column] = _matrix(metrics, column, n)[:, j]
                for diagnostic, (score, rho) in diag_values.items():
                    low, high = diag_bounds[diagnostic]
                    frame[f"d_{diagnostic}"], frame[f"rho_{diagnostic}"] = score[:, j], rho[:, j]
                    frame[f"{diagnostic}_lower"], frame[f"{diagnostic}_upper"] = low[j], high[j]
                    frame[f"rho_low_{diagnostic}"] = low[j] / high[j]
                    frame[f"typical_{diagnostic}"] = (score[:, j] >= low[j]) & (score[:, j] <= high[j])
                    frame[f"high_surprise_{diagnostic}"] = rho[:, j] > 1
                    frame[f"all_high_surprise_{diagnostic}"] = (rho > 1).all(1)
                    frame[f"at_least_one_not_high_surprise_{diagnostic}"] = (rho <= 1).any(1)
                frames.append(frame)
    if not frames:
        raise ValueError("No evaluation inputs were selected")
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(KEYS).any():
        raise ValueError("Duplicate dataset/candidate keys across splits")
    return data


def build_indirect_comparison(*, summary_multiplier, **kwargs):
    return build_comparison(method="indirect", summary_multiplier=summary_multiplier, **kwargs)


def build_direct_comparison(*, scoring_rule, **kwargs):
    return build_comparison(method="direct", scoring_rule=scoring_rule, **kwargs)


def binary_metrics(actual, predicted, score):
    """Exclude unknown actual labels/nonfinite scores; AUC uses average ranks."""
    actual = pd.Series(actual, dtype="boolean").reset_index(drop=True)
    predicted = np.asarray(predicted, dtype=bool)
    score = np.asarray(score, dtype=float)
    if predicted.shape != score.shape or predicted.shape != (len(actual),):
        raise ValueError("Actual labels, predictions, and scores must be aligned vectors")
    keep = actual.notna().to_numpy() & np.isfinite(score)
    truth = actual.loc[keep].to_numpy(bool)
    pred, values = predicted[keep], score[keep]
    tp, fp = int(np.sum(truth & pred)), int(np.sum(~truth & pred))
    fn, tn = int(np.sum(truth & ~pred)), int(np.sum(~truth & ~pred))
    positive, negative = int(truth.sum()), int((~truth).sum())
    auc = np.nan
    if positive and negative:
        auc = float((rankdata(values)[truth].sum() - positive * (positive + 1) / 2) / (positive * negative))
    return {"n_total": len(actual), "n": int(keep.sum()), "n_excluded": int((~keep).sum()),
            "n_error_positive": positive, "n_error_negative": negative,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "FNR": fn / positive if positive else np.nan,
            "FPR": fp / negative if negative else np.nan, "AUC": auc}


def _data_groups(data):
    yield "all", data
    for name, group in data.groupby("dataset_type", sort=False):
        yield name, group


def component_detection(data, *, method=None):
    """Report Gaussian-style component detection, separately by dataset type."""
    method = method or str(data.method.iloc[0])
    rows = []
    for dataset_type, group in _data_groups(data):
        for model, panel in group.groupby("candidate_model", sort=False):
            for metric in (("pmp",) if method == "direct" else ERROR_SPECS):
                _, lowcol, highcol = ERROR_SPECS[metric]
                prefix = "posterior" if metric == "posterior_mmd" else metric
                for diagnostic in DIAGNOSTICS:
                    rho = panel[f"rho_{diagnostic}"].to_numpy(float)
                    rows.append({"method": method, "dataset_type": dataset_type, "scope": "component",
                                 "candidate_model": model, "error_metric": metric, "diagnostic": diagnostic,
                                 "error_lower": panel[lowcol].iloc[0], "error_upper": panel[highcol].iloc[0],
                                 "rho_threshold": 1.,
                                 **binary_metrics(panel[f"{prefix}_error_positive"], rho > 1, rho)})
    return pd.DataFrame(rows)


def global_pmp_detection(data, *, method=None):
    """Report both 'any high' (Gaussian default) and 'all high' for indirect.

    The all-high score is MINIMUM rho across all four candidates. The any-high
    score is MAXIMUM rho. Direct diagnostics are shared across candidates.
    """
    method = method or str(data.method.iloc[0])
    rows = []
    rules = (("any_high_surprise", "max"), ("all_high_surprise", "min")) if method == "indirect" else (("shared", "max"),)
    for dataset_type, group in _data_groups(data):
        grouped = group.groupby(DATASET_KEYS, sort=False, dropna=False)
        if not grouped.size().eq(len(ASSUMED_MODELS)).all():
            raise ValueError("Global PMP detection requires every candidate for every dataset")
        actual = grouped["high_pmp_error_any"].agg(lambda x: x.iloc[0])
        for diagnostic in DIAGNOSTICS:
            for rule, aggregate in rules:
                rho = grouped[f"rho_{diagnostic}"].agg(aggregate).to_numpy(float)
                rows.append({"method": method, "dataset_type": dataset_type, "scope": "dataset",
                             "error_metric": "pmp_any_component", "diagnostic": diagnostic,
                             "diagnostic_rule": rule, "rho_threshold": 1.,
                             **binary_metrics(actual, rho > 1, rho)})
    return pd.DataFrame(rows)


def run_comparison(*, method, summary_multiplier=None, scoring_rule=None, overwrite=False, **kwargs):
    paths = comparison_paths(method=method, summary_multiplier=summary_multiplier,
                             scoring_rule=scoring_rule, result_dir=kwargs.get("result_dir", RESULT_DIR))
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Comparison already exists: {existing[0]}; use overwrite=True to rebuild")
    data = build_comparison(method=method, summary_multiplier=summary_multiplier, scoring_rule=scoring_rule, **kwargs)
    tables = {"data": data, "detection": component_detection(data),
              "global_pmp_detection": global_pmp_detection(data)}
    for name, frame in tables.items():
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(paths[name], index=False)
    print(f"Saved {method} comparison: {paths['data'].parent}")
    return tables


def run_indirect_comparison(*, summary_multiplier, **kwargs):
    return run_comparison(method="indirect", summary_multiplier=summary_multiplier, **kwargs)


def run_direct_comparison(*, scoring_rule, **kwargs):
    return run_comparison(method="direct", scoring_rule=scoring_rule, **kwargs)


def run_all_comparisons(*, summary_multipliers=(1, 2, 4), scoring_rules=SCORING_RULES, **kwargs):
    """Run only the configurations explicitly selected by the caller."""
    outputs = {}
    for multiplier in summary_multipliers:
        outputs[f"indirect_s{multiplier}D"] = run_indirect_comparison(summary_multiplier=multiplier, **kwargs)
    for rule in scoring_rules:
        outputs[f"direct_{rule}_s{DIRECT_SUMMARY_DIM}"] = run_direct_comparison(scoring_rule=rule, **kwargs)
    return outputs


def load_comparison(*, method, summary_multiplier=None, scoring_rule=None, result_dir=RESULT_DIR):
    paths = comparison_paths(method=method, summary_multiplier=summary_multiplier,
                             scoring_rule=scoring_rule, result_dir=result_dir)
    tables = {name: pd.read_csv(path, dtype={"id": str, "source_model": str}, keep_default_na=False)
              for name, path in paths.items()}
    data = tables["data"]
    for column in data:
        if (column.startswith(("typical_", "high_surprise_", "all_high_surprise_", "at_least_one_not_high_surprise_"))
                or column.endswith(("_valid", "_positive", "_available"))
                or column in ("well_specified", "model_matched", "high_pmp_error_any", "gold_converged", "all_gold_converged")):
            text = data[column].astype(str).str.lower()
            data[column] = pd.array(text.map({"true": True, "false": False, "": None}), dtype="boolean")
        elif column not in ("method", "scoring_rule", "split", *KEYS, "generating_model", "convergence_policy"):
            data[column] = pd.to_numeric(data[column].replace("", np.nan), errors="raise")
    for name in ("detection", "global_pmp_detection"):
        for column in tables[name]:
            if column not in ("method", "dataset_type", "scope", "candidate_model", "error_metric", "diagnostic", "diagnostic_rule"):
                tables[name][column] = pd.to_numeric(tables[name][column].replace("", np.nan), errors="raise")
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("indirect", "direct", "both"), default="both")
    parser.add_argument("--summary-multipliers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--scoring-rules", nargs="+", choices=SCORING_RULES, default=list(SCORING_RULES))
    parser.add_argument("--splits", nargs="+", choices=("simulated", "empirical", "benchmark"), default=["simulated", "empirical"])
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--convergence-policy", choices=("available", "strict", "all"), default="available")
    parser.add_argument("--allow-missing-empirical", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_all_comparisons(summary_multipliers=args.summary_multipliers if args.method != "direct" else (),
                       scoring_rules=args.scoring_rules if args.method != "indirect" else (),
                       splits=args.splits, result_dir=args.result_dir, convergence_policy=args.convergence_policy,
                       allow_missing_empirical=args.allow_missing_empirical, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
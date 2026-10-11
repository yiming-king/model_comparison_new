"""Calibrate and evaluate RDM diagnostics in learned summary space.

The reference and calibration banks are fixed prior-predictive datasets.  The
four score definitions match the Gaussian example's diagnostic implementation.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from pathlib import Path

import keras
import numpy as np

from ...gaussian.analysis.diagnostic import (
    DIAGNOSTICS,
    central_interval,
    compute_non_density_diagnostics,
    density_nll,
    diagnostic_scores,
    fit_density_flow,
    fit_reference,
)
from ..approximators.config import DIRECT_SCORING_RULES, SUMMARY_MULTIPLIERS, TrainingConfig, checkpoint_path
from ..config import ASSUMED_MODELS, NUM_TRIALS, PARAM_DIMS, RESULT_DIR, SIMULATED_DATASETS
from ..datasets import load_benchmark_dataset, load_diagnostic_dataset, load_empirical_dataset, load_simulated_dataset
from .inference import DIRECT_SUMMARY_DIM, data_conditions, load_approximator


def diagnostic_root(*, method, summary_multiplier=None, scoring_rule=None):
    if method == "indirect" and summary_multiplier in SUMMARY_MULTIPLIERS:
        name = f"indirect_s{summary_multiplier}D"
    elif method == "direct" and scoring_rule in DIRECT_SCORING_RULES:
        name = f"direct_{scoring_rule}_s{DIRECT_SUMMARY_DIM}"
    else:
        raise ValueError("Choose an indirect summary multiplier or a direct scoring rule")
    return RESULT_DIR / "diagnostics" / f"n{NUM_TRIALS}" / name


def reference_path(*, method, model=None, summary_multiplier=None, scoring_rule=None):
    root = diagnostic_root(method=method, summary_multiplier=summary_multiplier, scoring_rule=scoring_rule)
    return root / "reference" / (f"{model}.npz" if method == "indirect" else "mixed.npz")


def density_path(*, method, model=None, summary_multiplier=None, scoring_rule=None):
    return reference_path(
        method=method, model=model, summary_multiplier=summary_multiplier, scoring_rule=scoring_rule
    ).with_suffix(".keras")


def threshold_path(*, method, summary_multiplier=None, scoring_rule=None):
    return diagnostic_root(method=method, summary_multiplier=summary_multiplier, scoring_rule=scoring_rule) / "thresholds.npz"


def diagnostic_path(*, method, split, source, summary_multiplier=None, scoring_rule=None):
    return diagnostic_root(
        method=method, summary_multiplier=summary_multiplier, scoring_rule=scoring_rule
    ) / split / f"{source}.npz"


def _save(values, path, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _datasets(split, source):
    if split in ("reference", "calibration"):
        bank = load_diagnostic_dataset(split, source)
        return np.asarray(bank["data"], np.float32), np.asarray(bank["ids"])
    if split == "benchmark":
        result = load_benchmark_dataset(source)
    elif split == "simulated":
        result = load_simulated_dataset(source)
    elif split == "empirical":
        result = load_empirical_dataset(source)
    else:
        raise ValueError(f"Unknown dataset split: {split}")
    return np.asarray(result["data"], np.float32), np.asarray(result["ids"])


def _summary(approximator, data):
    return np.asarray(approximator.summarize(conditions=data_conditions(data)), dtype=np.float64)


def _indirect_approximators(summary_multiplier):
    return {
        model: load_approximator(checkpoint_path(TrainingConfig(summary_dim=PARAM_DIMS[model] * summary_multiplier), model=model))
        for model in ASSUMED_MODELS
    }


def _direct_approximator(scoring_rule):
    return load_approximator(checkpoint_path(TrainingConfig(summary_dim=DIRECT_SUMMARY_DIM), scoring_rule=scoring_rule))


def fit_indirect_references(*, summary_multiplier, density_epochs=250, density_batch_size=128,
                            seed=2025, overwrite=False):
    """Fit one summary reference and density flow per candidate model."""
    approximators = _indirect_approximators(summary_multiplier)
    for j, model in enumerate(ASSUMED_MODELS):
        ref_path = reference_path(method="indirect", model=model, summary_multiplier=summary_multiplier)
        flow_path = density_path(method="indirect", model=model, summary_multiplier=summary_multiplier)
        if ref_path.exists() and flow_path.exists() and not overwrite:
            continue
        if (ref_path.exists() or flow_path.exists()) and not overwrite:
            raise FileExistsError(f"Partial diagnostic reference for {model}: {ref_path.parent}")
        data, _ = _datasets("reference", model)
        z = _summary(approximators[model], data)
        reference = fit_reference(z, seed=seed + j)
        flow = fit_density_flow(z, epochs=density_epochs, batch_size=density_batch_size, seed=seed + j)
        _save(reference, ref_path, overwrite=overwrite)
        flow.save(flow_path, overwrite=overwrite)
        print(f"Saved indirect diagnostic reference: {model}, S={summary_multiplier}D")


def calibrate_indirect(*, summary_multiplier, coverage=0.90, overwrite=False):
    """Calibrate score intervals from matching-model calibration banks."""
    output = threshold_path(method="indirect", summary_multiplier=summary_multiplier)
    if output.exists() and not overwrite:
        return _load(output)
    approximators = _indirect_approximators(summary_multiplier)
    results = {"candidate_models": np.asarray(ASSUMED_MODELS), "density_center": np.empty(4)}
    for diagnostic in DIAGNOSTICS:
        results[f"d_{diagnostic}_lower"] = np.empty(4)
        results[f"d_{diagnostic}_upper"] = np.empty(4)
    for j, model in enumerate(ASSUMED_MODELS):
        reference = _load(reference_path(method="indirect", model=model, summary_multiplier=summary_multiplier))
        flow = keras.saving.load_model(density_path(method="indirect", model=model, summary_multiplier=summary_multiplier))
        data, _ = _datasets("calibration", model)
        z = _summary(approximators[model], data)
        scores = compute_non_density_diagnostics(z, reference)
        nll = density_nll(flow, z)
        results["density_center"][j] = nll.mean()
        scores["d_density"] = nll - nll.mean()
        for diagnostic in DIAGNOSTICS:
            low, high = central_interval(scores[f"d_{diagnostic}"], coverage=coverage)
            results[f"d_{diagnostic}_lower"][j] = low
            results[f"d_{diagnostic}_upper"][j] = high
    _save(results, output, overwrite=overwrite)
    print(f"Saved indirect diagnostic thresholds: {output}")
    return results


def run_indirect_diagnostics(*, split, source, summary_multiplier, overwrite=False):
    """Evaluate all four diagnostic scores for one dataset group."""
    output = diagnostic_path(method="indirect", split=split, source=source, summary_multiplier=summary_multiplier)
    if output.exists() and not overwrite:
        return
    data, ids = _datasets(split, source)
    approximators = _indirect_approximators(summary_multiplier)
    thresholds = _load(threshold_path(method="indirect", summary_multiplier=summary_multiplier))
    results = {"id": ids, "candidate_models": np.asarray(ASSUMED_MODELS)}
    scores_by_metric = {diagnostic: np.empty((len(ids), 4)) for diagnostic in DIAGNOSTICS}
    for j, model in enumerate(ASSUMED_MODELS):
        reference = _load(reference_path(method="indirect", model=model, summary_multiplier=summary_multiplier))
        flow = keras.saving.load_model(density_path(method="indirect", model=model, summary_multiplier=summary_multiplier))
        z = _summary(approximators[model], data)
        scores = diagnostic_scores(z, reference, flow, thresholds["density_center"][j])
        for diagnostic in DIAGNOSTICS:
            scores_by_metric[diagnostic][:, j] = scores[f"d_{diagnostic}"]
    for diagnostic, values in scores_by_metric.items():
        low = thresholds[f"d_{diagnostic}_lower"][None, :]
        high = thresholds[f"d_{diagnostic}_upper"][None, :]
        results[f"d_{diagnostic}"] = values
        results[f"rho_{diagnostic}"] = values / high
        results[f"typical_{diagnostic}"] = (values >= low) & (values <= high)
    _save(results, output, overwrite=overwrite)
    print(f"Saved indirect diagnostics: {output}")


def _direct_mixed_summaries(approximator, split):
    return np.concatenate([
        _summary(approximator, _datasets(split, model)[0]) for model in ASSUMED_MODELS
    ], axis=0)


def fit_direct_reference(*, scoring_rule, density_epochs=250, density_batch_size=128,
                         seed=2025, overwrite=False):
    """Fit one reference to the equal-model prior-predictive mixture."""
    ref_path = reference_path(method="direct", scoring_rule=scoring_rule)
    flow_path = density_path(method="direct", scoring_rule=scoring_rule)
    if ref_path.exists() and flow_path.exists() and not overwrite:
        return
    if (ref_path.exists() or flow_path.exists()) and not overwrite:
        raise FileExistsError(f"Partial direct diagnostic reference: {ref_path.parent}")
    z = _direct_mixed_summaries(_direct_approximator(scoring_rule), "reference")
    reference = fit_reference(z, seed=seed)
    flow = fit_density_flow(z, epochs=density_epochs, batch_size=density_batch_size, seed=seed)
    _save(reference, ref_path, overwrite=overwrite)
    flow.save(flow_path, overwrite=overwrite)
    print(f"Saved direct diagnostic reference: {scoring_rule}")


def calibrate_direct(*, scoring_rule, coverage=0.90, overwrite=False):
    output = threshold_path(method="direct", scoring_rule=scoring_rule)
    if output.exists() and not overwrite:
        return _load(output)
    reference = _load(reference_path(method="direct", scoring_rule=scoring_rule))
    flow = keras.saving.load_model(density_path(method="direct", scoring_rule=scoring_rule))
    z = _direct_mixed_summaries(_direct_approximator(scoring_rule), "calibration")
    scores = compute_non_density_diagnostics(z, reference)
    nll = density_nll(flow, z)
    results = {"density_center": np.asarray(nll.mean())}
    scores["d_density"] = nll - nll.mean()
    for diagnostic in DIAGNOSTICS:
        low, high = central_interval(scores[f"d_{diagnostic}"], coverage=coverage)
        results[f"d_{diagnostic}_lower"] = np.asarray(low)
        results[f"d_{diagnostic}_upper"] = np.asarray(high)
    _save(results, output, overwrite=overwrite)
    print(f"Saved direct diagnostic thresholds: {output}")
    return results


def run_direct_diagnostics(*, split, source, scoring_rule, overwrite=False):
    output = diagnostic_path(method="direct", split=split, source=source, scoring_rule=scoring_rule)
    if output.exists() and not overwrite:
        return
    data, ids = _datasets(split, source)
    z = _summary(_direct_approximator(scoring_rule), data)
    reference = _load(reference_path(method="direct", scoring_rule=scoring_rule))
    flow = keras.saving.load_model(density_path(method="direct", scoring_rule=scoring_rule))
    thresholds = _load(threshold_path(method="direct", scoring_rule=scoring_rule))
    scores = diagnostic_scores(z, reference, flow, float(thresholds["density_center"]))
    results = {"id": ids}
    for diagnostic in DIAGNOSTICS:
        values = scores[f"d_{diagnostic}"]
        low = float(thresholds[f"d_{diagnostic}_lower"])
        high = float(thresholds[f"d_{diagnostic}_upper"])
        results[f"d_{diagnostic}"] = values
        results[f"rho_{diagnostic}"] = values / high
        results[f"typical_{diagnostic}"] = (values >= low) & (values <= high)
    _save(results, output, overwrite=overwrite)
    print(f"Saved direct diagnostics: {output}")


def run_all_diagnostics(*, splits=("simulated", "empirical"), density_epochs=250, overwrite=False):
    """Build references, thresholds, and requested split diagnostics for all runs."""
    sources = {"simulated": SIMULATED_DATASETS, "benchmark": ASSUMED_MODELS, "empirical": ("empirical",)}
    for summary_multiplier in SUMMARY_MULTIPLIERS:
        fit_indirect_references(summary_multiplier=summary_multiplier, density_epochs=density_epochs, overwrite=overwrite)
        calibrate_indirect(summary_multiplier=summary_multiplier, overwrite=overwrite)
        for split in splits:
            for source in sources[split]:
                run_indirect_diagnostics(
                    split=split, source=source, summary_multiplier=summary_multiplier, overwrite=overwrite
                )
    for scoring_rule in DIRECT_SCORING_RULES:
        fit_direct_reference(scoring_rule=scoring_rule, density_epochs=density_epochs, overwrite=overwrite)
        calibrate_direct(scoring_rule=scoring_rule, overwrite=overwrite)
        for split in splits:
            for source in sources[split]:
                run_direct_diagnostics(split=split, source=source, scoring_rule=scoring_rule, overwrite=overwrite)

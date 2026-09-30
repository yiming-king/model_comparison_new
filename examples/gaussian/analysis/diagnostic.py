"""Summary-space diagnostics for the Gaussian case study."""
# fmt: off

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from pathlib import Path

import bayesflow as bf
import keras
import numpy as np
from sklearn.covariance import LedoitWolf

from ..approximators.config import TrainingConfig, checkpoint_path
from ..config import ASSUMED_MODELS, SOURCE_MODELS, DATASET_DIR, RESULT_DIR
from ..datasets.datasets import load_datasets
from .inference import load_approximator

DIAGNOSTIC_DIR = RESULT_DIR / "diagnostics"

DIAGNOSTICS = ("l2", "linf", "density", "mmd")

# ---------------------------------------------------------------------
# Paths / IO
# ---------------------------------------------------------------------

def diagnostic_root(*, method, num_obs, summary_dim, scoring_rule=None):
    """Return the root directory for one diagnostic configuration."""
    if method == "indirect":
        name = f"indirect_s{summary_dim}"
    elif method == "direct":
        if scoring_rule is None:
            raise ValueError("scoring_rule is required for direct diagnostics")
        name = f"direct_{scoring_rule}_s{summary_dim}"
    else:
        raise ValueError("method must be 'indirect' or 'direct'")
    return DIAGNOSTIC_DIR / f"n{num_obs}" / name

def indirect_reference_path(*, num_obs, summary_dim, model):
    return diagnostic_root(method="indirect", num_obs=num_obs, summary_dim=summary_dim) / "reference" / f"{model}.npz"

def indirect_density_path(*, num_obs, summary_dim, model):
    return (
        diagnostic_root(method="indirect", num_obs=num_obs, summary_dim=summary_dim)
        / "reference"
        / f"{model}_density.keras"
    )

def direct_reference_path(*, num_obs, summary_dim, scoring_rule):
    return (
        diagnostic_root(method="direct", num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
        / "reference.npz"
    )

def direct_density_path(*, num_obs, summary_dim, scoring_rule):
    return (
        diagnostic_root(method="direct", num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
        / "density.keras"
    )

def diagnostic_threshold_path(*, method, num_obs, summary_dim, scoring_rule=None):
    return (
        diagnostic_root(method=method, num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
        / "thresholds.npz"
    )

def diagnostic_result_path(*, method, split, source_model, num_obs, summary_dim, scoring_rule=None):
    return (
        diagnostic_root(method=method, num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
        / split
        / f"{source_model}.npz"
    )

def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def save_npz(values, path, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)

# ---------------------------------------------------------------------
# Dataset / summary-network helpers
# ---------------------------------------------------------------------

def load_x(*, split, source_model, num_obs):
    """Load observation arrays for one fixed dataset file."""
    path = DATASET_DIR / f"n{num_obs}" / split / f"{source_model}.npz"
    datasets = load_datasets(path)
    return (np.asarray(datasets["x"], dtype=np.float32), datasets["id"], datasets["source_model"])

def load_indirect_approximators(*, num_obs, summary_dim, num_dims=20):
    """Load the four model-specific NPE approximators."""
    config = TrainingConfig(
        num_dims=num_dims, num_obs=num_obs, summary_dim=summary_dim, summary_base_distribution="normal"
    )
    return {model: load_approximator(checkpoint_path(config, model=model)) for model in ASSUMED_MODELS}

def load_direct_approximator(*, num_obs, summary_dim, scoring_rule, num_dims=20):
    """Load one direct model-comparison approximator."""
    config = TrainingConfig(num_dims=num_dims, num_obs=num_obs, summary_dim=summary_dim)
    return load_approximator(checkpoint_path(config, scoring_rule=scoring_rule))

def summarize(approximator, x):
    """Return learned summary vectors."""
    return np.asarray(approximator.summarize(conditions={"x": np.asarray(x, dtype=np.float32)}), dtype=np.float64)

def indirect_summaries(approximators, x):
    """
    Compute summaries from all four indirect NPE networks.

    Output:
        (B, 4, summary_dim)
    """
    return np.stack([summarize(approximators[model], x) for model in ASSUMED_MODELS], axis=1)

# ---------------------------------------------------------------------
# L2 / Linf reference
# ---------------------------------------------------------------------

def fit_moment_reference(summaries):
    """Fit mean and Ledoit-Wolf covariance."""
    summaries = np.asarray(summaries, dtype=np.float64)
    covariance = LedoitWolf().fit(summaries)
    mean = covariance.location_
    chol = np.linalg.cholesky(covariance.covariance_)
    return mean, chol

def whiten_summaries(summaries, mean, chol):
    """Whiten summary vectors."""
    summaries = np.asarray(summaries, dtype=np.float64)
    centered = summaries - mean
    return np.linalg.solve(chol, centered.T).T

def l2_distance(summaries, mean, chol):
    """
    Dimension-normalized L2 diagnostic.

    D_L2 = ||z_white||_2 / sqrt(S)
    """
    whitened = whiten_summaries(summaries, mean, chol)
    summary_dim = whitened.shape[1]
    return np.linalg.norm(whitened, axis=1) / np.sqrt(summary_dim)

def linf_distance(summaries, mean, chol):
    """
    Dimension-normalized Linf diagnostic.

    D_Linf = ||z_white||_inf / sqrt(2 log S)
    """
    whitened = whiten_summaries(summaries, mean, chol)
    summary_dim = whitened.shape[1]
    scale = np.sqrt(2.0 * np.log(summary_dim)) if summary_dim > 1 else 1.0
    return np.max(np.abs(whitened), axis=1) / scale

# ---------------------------------------------------------------------
# Kernel / MMD reference
# ---------------------------------------------------------------------

def mmd_bandwidth2(summaries, *, max_pairs=500_000, seed=2025):
    """
    Median heuristic using positive squared pairwise distances.
    """
    summaries = np.asarray(summaries, dtype=np.float64)
    n = len(summaries)
    if n < 2:
        raise ValueError("At least two reference summaries are required")
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        diff = summaries[:, None, :] - summaries[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        positive = dist2[dist2 > 0.0]
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n - 1, size=max_pairs)
        j = j + (j >= i)
        diff = summaries[i] - summaries[j]
        positive = np.sum(diff * diff, axis=1)
    if len(positive) == 0:
        return 1.0
    return max(float(np.median(positive)), 1e-8)

def rbf_kernel_mean(x, y, bandwidth2, *, chunk_size=256):
    """
    Exact mean RBF kernel value with bounded memory.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    total = 0.0
    count = 0
    for x_start in range(0, len(x), chunk_size):
        x_block = x[x_start : x_start + chunk_size]
        for y_start in range(0, len(y), chunk_size):
            y_block = y[y_start : y_start + chunk_size]
            diff = x_block[:, None, :] - y_block[None, :, :]
            dist2 = np.sum(diff * diff, axis=-1)
            kernel = np.exp(-dist2 / (2.0 * bandwidth2))
            total += float(kernel.sum())
            count += len(x_block) * len(y_block)
    return total / count

def cross_kernel_mean_per_point(summaries, reference, bandwidth2, *, summary_chunk=256, reference_chunk=512):
    """
    E_y[k(z, y)] for each z.
    """
    summaries = np.asarray(summaries, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    result = np.empty(len(summaries), dtype=np.float64)
    for start in range(0, len(summaries), summary_chunk):
        block = summaries[start : start + summary_chunk]
        kernel_sum = np.zeros(len(block), dtype=np.float64)
        for ref_start in range(0, len(reference), reference_chunk):
            ref_block = reference[ref_start : ref_start + reference_chunk]
            diff = block[:, None, :] - ref_block[None, :, :]
            dist2 = np.sum(diff * diff, axis=-1)
            kernel_sum += np.exp(-dist2 / (2.0 * bandwidth2)).sum(axis=1)
        result[start : start + len(block)] = kernel_sum / len(reference)
    return result

def mmd_distance(summaries, reference, bandwidth2, reference_kernel_mean):
    """
    RBF-MMD between each summary point and the reference distribution.

    MMD^2(delta_z, P_ref)
        = 1
        + E[k(Y,Y')]
        - 2 E[k(z,Y)]
    """
    cross_mean = cross_kernel_mean_per_point(summaries, reference, bandwidth2)
    mmd2 = 1.0 + float(reference_kernel_mean) - 2.0 * cross_mean
    return np.sqrt(np.maximum(mmd2, 0.0))

# ---------------------------------------------------------------------
# Density diagnostic
# ---------------------------------------------------------------------

def fit_density_flow(
    summaries,
    *,
    epochs=250,
    batch_size=128,
    depth=8,
    widths=(256, 256, 256),
    learning_rate=5e-4,
    patience=20,
    start_from_epoch=50,
    seed=2025,
):
    """Fit an unconditional flow to reference summaries."""
    keras.utils.set_random_seed(seed)
    summaries = np.asarray(summaries, dtype=np.float32)
    adapter = bf.Adapter().convert_dtype("float64", "float32").rename("z", "inference_variables")
    dataset = bf.datasets.OfflineDataset(data={"z": summaries}, batch_size=batch_size, adapter=adapter, shuffle=True)
    flow = bf.approximators.ContinuousApproximator(
        inference_network=bf.networks.CouplingFlow(
            depth=depth, subnet_kwargs={"widths": widths, "activation": "mish", "norm": "layer"}
        ),
        standardize="inference_variables",
    )
    learning_rate_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate, decay_steps=(epochs * dataset.num_batches), alpha=1e-6
    )
    flow.compile(optimizer=keras.optimizers.Adam(learning_rate_schedule))
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="loss", patience=patience, start_from_epoch=start_from_epoch, restore_best_weights=True
        )
    ]
    flow.fit(dataset=dataset, epochs=epochs, callbacks=callbacks, verbose=0)
    return flow

def density_nll(flow, summaries):
    """Negative log density under the fitted reference flow."""
    summaries = np.asarray(summaries, dtype=np.float32)
    log_prob = flow.log_prob({"inference_variables": summaries})
    return -np.asarray(log_prob, dtype=np.float64).reshape(-1)

# ---------------------------------------------------------------------
# Reference construction
# ---------------------------------------------------------------------

def fit_reference(summaries, *, seed=2025):
    """
    Fit all non-density reference quantities.
    """
    summaries = np.asarray(summaries, dtype=np.float64)
    mean, chol = fit_moment_reference(summaries)
    bandwidth2 = mmd_bandwidth2(summaries, seed=seed)
    reference_kernel_mean = rbf_kernel_mean(summaries, summaries, bandwidth2)
    return {
        "mean": mean,
        "chol": chol,
        "reference_summaries": (summaries.astype(np.float32)),
        "mmd_bandwidth2": (bandwidth2),
        "mmd_reference_kernel_mean": (reference_kernel_mean),
    }

def compute_non_density_diagnostics(summaries, reference):
    """Compute L2, Linf, and kernel diagnostics."""
    return {
        "d_l2": l2_distance(summaries, reference["mean"], reference["chol"]),
        "d_linf": linf_distance(summaries, reference["mean"], reference["chol"]),
        "d_mmd": mmd_distance(
            summaries,
            reference["reference_summaries"],
            float(reference["mmd_bandwidth2"]),
            float(reference["mmd_reference_kernel_mean"]),
        ),
    }

# ---------------------------------------------------------------------
# Diagnostic thresholds
# ---------------------------------------------------------------------

def central_interval(values, *, coverage=0.90):
    """Central empirical interval."""
    alpha = (1.0 - coverage) / 2.0
    values = np.asarray(values, dtype=np.float64)
    return (float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha)))

def diagnostic_scores(summaries, reference, density_flow, density_center):
    """
    Compute all four diagnostics.

    D_density is centered negative log density:
        -log q(z) - mean_cal[-log q(z)]
    """
    scores = compute_non_density_diagnostics(summaries, reference)
    scores["d_density"] = density_nll(density_flow, summaries) - density_center
    return scores

# ---------------------------------------------------------------------
# INDIRECT: references
# ---------------------------------------------------------------------

def fit_indirect_references(
    *, num_obs, summary_dim, density_epochs=250, density_batch_size=128, seed=2025, overwrite=False
):
    """
    Fit one model-specific reference distribution per NPE.

    M1 network uses reference datasets generated by M1,
    M2 network uses M2 reference datasets, etc.
    """
    approximators = load_indirect_approximators(num_obs=num_obs, summary_dim=summary_dim)
    for model_index, model in enumerate(ASSUMED_MODELS):
        ref_path = indirect_reference_path(num_obs=num_obs, summary_dim=summary_dim, model=model)
        density_path = indirect_density_path(num_obs=num_obs, summary_dim=summary_dim, model=model)
        if (ref_path.exists() or density_path.exists()) and not overwrite:
            raise FileExistsError(f"Indirect reference exists: {model}")
        x, _, _ = load_x(split="reference", source_model=model, num_obs=num_obs)
        z = summarize(approximators[model], x)
        reference = fit_reference(z, seed=seed + model_index)
        save_npz(reference, ref_path, overwrite=overwrite)
        density_path.parent.mkdir(parents=True, exist_ok=True)
        density_flow = fit_density_flow(
            z, epochs=density_epochs, batch_size=density_batch_size, seed=seed + model_index
        )
        density_flow.save(density_path, overwrite=True)
        print(f"Saved indirect reference: {model}")

# ---------------------------------------------------------------------
# INDIRECT: calibration
# ---------------------------------------------------------------------

def calibrate_indirect_diagnostics(*, num_obs, summary_dim, coverage=0.90, overwrite=False):
    """
    Calibrate model-specific diagnostic intervals.

    Each model uses matching well-specified calibration datasets.
    """
    approximators = load_indirect_approximators(num_obs=num_obs, summary_dim=summary_dim)
    num_models = len(ASSUMED_MODELS)
    lower = {metric: np.empty(num_models, dtype=np.float64) for metric in DIAGNOSTICS}
    upper = {metric: np.empty(num_models, dtype=np.float64) for metric in DIAGNOSTICS}
    density_center = np.empty(num_models, dtype=np.float64)
    for model_index, model in enumerate(ASSUMED_MODELS):
        reference = load_npz(indirect_reference_path(num_obs=num_obs, summary_dim=summary_dim, model=model))
        density_flow = keras.saving.load_model(
            indirect_density_path(num_obs=num_obs, summary_dim=summary_dim, model=model)
        )
        x, _, _ = load_x(split="calibration", source_model=model, num_obs=num_obs)
        z = summarize(approximators[model], x)
        non_density = compute_non_density_diagnostics(z, reference)
        calibration_nll = density_nll(density_flow, z)
        density_center[model_index] = calibration_nll.mean()
        scores = {**non_density, "d_density": (calibration_nll - density_center[model_index])}
        for metric in DIAGNOSTICS:
            low, high = central_interval(scores[f"d_{metric}"], coverage=coverage)
            lower[metric][model_index] = low
            upper[metric][model_index] = high
    results = {"candidate_models": (np.asarray(ASSUMED_MODELS)), "density_center": (density_center)}
    for metric in DIAGNOSTICS:
        results[f"d_{metric}_lower"] = lower[metric]
        results[f"d_{metric}_upper"] = upper[metric]
    output_path = diagnostic_threshold_path(method="indirect", num_obs=num_obs, summary_dim=summary_dim)
    save_npz(results, output_path, overwrite=overwrite)
    print(f"Saved indirect diagnostic thresholds: {output_path}")
    return results

# ---------------------------------------------------------------------
# INDIRECT: evaluation
# ---------------------------------------------------------------------

def compute_indirect_diagnostics(*, x, ids, source_model, approximators, references, density_flows, thresholds):
    """Compute all diagnostics for one source-model dataset file."""
    summaries = indirect_summaries(approximators, x)
    num_datasets = len(x)
    num_models = len(ASSUMED_MODELS)
    values = {metric: np.empty((num_datasets, num_models), dtype=np.float64) for metric in DIAGNOSTICS}
    for model_index, model in enumerate(ASSUMED_MODELS):
        scores = diagnostic_scores(
            summaries[:, model_index, :],
            references[model],
            density_flows[model],
            thresholds["density_center"][model_index],
        )
        for metric in DIAGNOSTICS:
            values[metric][:, model_index] = scores[f"d_{metric}"]
    result = {
        "id": ids,
        "source_model": source_model,
        "candidate_models": np.asarray(ASSUMED_MODELS),
        "summaries": summaries,
    }
    for metric in DIAGNOSTICS:
        score = values[metric]
        low = thresholds[f"d_{metric}_lower"][None, :]
        high = thresholds[f"d_{metric}_upper"][None, :]
        result[f"d_{metric}"] = score
        result[f"rho_{metric}"] = score / high
        result[f"typical_{metric}"] = (score >= low) & (score <= high)
    return result

def run_indirect_diagnostics(*, split, num_obs, summary_dim, sources=None, overwrite=False):
    """Compute indirect diagnostics for a complete dataset split."""
    if sources is None:
        sources = SOURCE_MODELS if split == "simulated" else ASSUMED_MODELS
    approximators = load_indirect_approximators(num_obs=num_obs, summary_dim=summary_dim)
    references = {
        model: load_npz(indirect_reference_path(num_obs=num_obs, summary_dim=summary_dim, model=model))
        for model in ASSUMED_MODELS
    }
    density_flows = {
        model: keras.saving.load_model(indirect_density_path(num_obs=num_obs, summary_dim=summary_dim, model=model))
        for model in ASSUMED_MODELS
    }
    thresholds = load_npz(diagnostic_threshold_path(method="indirect", num_obs=num_obs, summary_dim=summary_dim))
    for source in sources:
        x, ids, source_model = load_x(split=split, source_model=source, num_obs=num_obs)
        results = compute_indirect_diagnostics(
            x=x,
            ids=ids,
            source_model=source_model,
            approximators=approximators,
            references=references,
            density_flows=density_flows,
            thresholds=thresholds,
        )
        output_path = diagnostic_result_path(
            method="indirect", split=split, source_model=source, num_obs=num_obs, summary_dim=summary_dim
        )
        save_npz(results, output_path, overwrite=overwrite)
        print(f"Saved indirect diagnostics: {source}")

# ---------------------------------------------------------------------
# DIRECT: reference
# ---------------------------------------------------------------------

def direct_mixed_summaries(approximator, *, split, num_obs):
    """
    Combine M1-M4 summaries into one equal-model mixture.

    Since each stored source file has the same number of datasets,
    concatenation represents the equal model prior used in training.
    """
    summaries = []
    for model in ASSUMED_MODELS:
        x, _, _ = load_x(split=split, source_model=model, num_obs=num_obs)
        summaries.append(summarize(approximator, x))
    return np.concatenate(summaries, axis=0)

def fit_direct_reference(
    *, num_obs, summary_dim, scoring_rule, density_epochs=250, density_batch_size=128, seed=2025, overwrite=False
):
    """
    Fit one shared mixed-model reference for direct model comparison.
    """
    ref_path = direct_reference_path(num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
    density_path = direct_density_path(num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
    if (ref_path.exists() or density_path.exists()) and not overwrite:
        raise FileExistsError("Direct diagnostic reference already exists")
    approximator = load_direct_approximator(num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
    z = direct_mixed_summaries(approximator, split="reference", num_obs=num_obs)
    reference = fit_reference(z, seed=seed)
    save_npz(reference, ref_path, overwrite=overwrite)
    density_flow = fit_density_flow(z, epochs=density_epochs, batch_size=density_batch_size, seed=seed)
    density_path.parent.mkdir(parents=True, exist_ok=True)
    density_flow.save(density_path, overwrite=True)
    print(f"Saved direct reference: {ref_path}")

# ---------------------------------------------------------------------
# DIRECT: calibration
# ---------------------------------------------------------------------

def calibrate_direct_diagnostics(*, num_obs, summary_dim, scoring_rule, coverage=0.90, overwrite=False):
    """
    Calibrate one shared diagnostic interval from mixed M1-M4 data.
    """
    approximator = load_direct_approximator(num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
    reference = load_npz(direct_reference_path(num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule))
    density_flow = keras.saving.load_model(
        direct_density_path(num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
    )
    z = direct_mixed_summaries(approximator, split="calibration", num_obs=num_obs)
    non_density = compute_non_density_diagnostics(z, reference)
    calibration_nll = density_nll(density_flow, z)
    density_center = float(calibration_nll.mean())
    scores = {**non_density, "d_density": (calibration_nll - density_center)}
    results = {"density_center": (density_center)}
    for metric in DIAGNOSTICS:
        low, high = central_interval(scores[f"d_{metric}"], coverage=coverage)
        results[f"d_{metric}_lower"] = low
        results[f"d_{metric}_upper"] = high
    output_path = diagnostic_threshold_path(
        method="direct", num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule
    )
    save_npz(results, output_path, overwrite=overwrite)
    print(f"Saved direct diagnostic thresholds: {output_path}")
    return results

# ---------------------------------------------------------------------
# DIRECT: evaluation
# ---------------------------------------------------------------------

def compute_direct_diagnostics(*, summaries, ids, source_model, reference, density_flow, thresholds):
    """Compute shared direct diagnostic values."""
    scores = diagnostic_scores(summaries, reference, density_flow, float(thresholds["density_center"]))
    result = {"id": ids, "source_model": source_model, "summaries": summaries}
    for metric in DIAGNOSTICS:
        score = scores[f"d_{metric}"]
        low = float(thresholds[f"d_{metric}_lower"])
        high = float(thresholds[f"d_{metric}_upper"])
        result[f"d_{metric}"] = score
        result[f"rho_{metric}"] = score / high
        result[f"typical_{metric}"] = (score >= low) & (score <= high)
    return result

def run_direct_diagnostics(*, split, num_obs, summary_dim, scoring_rule, sources=None, overwrite=False):
    """Compute direct diagnostics for a complete dataset split."""
    if sources is None:
        sources = SOURCE_MODELS if split == "simulated" else ASSUMED_MODELS
    approximator = load_direct_approximator(num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
    reference = load_npz(direct_reference_path(num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule))
    density_flow = keras.saving.load_model(
        direct_density_path(num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
    )
    thresholds = load_npz(
        diagnostic_threshold_path(method="direct", num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
    )
    for source in sources:
        x, ids, source_model = load_x(split=split, source_model=source, num_obs=num_obs)
        z = summarize(approximator, x)
        results = compute_direct_diagnostics(
            summaries=z,
            ids=ids,
            source_model=source_model,
            reference=reference,
            density_flow=density_flow,
            thresholds=thresholds,
        )
        output_path = diagnostic_result_path(
            method="direct",
            split=split,
            source_model=source,
            num_obs=num_obs,
            summary_dim=summary_dim,
            scoring_rule=scoring_rule,
        )
        save_npz(results, output_path, overwrite=overwrite)
        print(f"Saved direct diagnostics: {source}")

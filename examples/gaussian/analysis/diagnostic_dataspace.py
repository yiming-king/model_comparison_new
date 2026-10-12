"""Four model-specific Gaussian diagnostics on vec(y), not on learned summaries.

Raw datasets remain in results/datasets. The stats cache stores fitted parameters,
not reference observations. No NPE checkpoint, posterior sample, or summary is used.
BayesFlow/Keras are imported lazily, only for the separate diagnostic density flows.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist, pdist
from sklearn.covariance import LedoitWolf

from ..config import ASSUMED_MODELS, SOURCE_MODELS, DATASET_DIR, RESULT_DIR

DIAGNOSTICS = ("l2", "linf", "density", "mmd")
SCHEMA_VERSION = 1


def diagnostic_root(*, num_obs):
    return RESULT_DIR / "data_space" / "diagnostics" / f"n{int(num_obs)}"


def stats_path(*, num_obs, model):
    return diagnostic_root(num_obs=num_obs) / "reference_stats" / f"{model}_stats.npz"


def density_path(*, num_obs, model):
    return diagnostic_root(num_obs=num_obs) / "density" / f"{model}.keras"


def diagnostic_threshold_path(*, num_obs):
    return diagnostic_root(num_obs=num_obs) / "thresholds.npz"


def diagnostic_result_path(*, num_obs, split, source_model):
    return diagnostic_root(num_obs=num_obs) / split / f"{source_model}.npz"


def load_npz(path, keys=None):
    """Read selected arrays; do not load huge posterior arrays just to read metadata."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: archive[key] for key in (archive.files if keys is None else keys)}


def save_npz(values, path, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **values)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_vectors(*, num_obs, split, source_model):
    """One complete (N,D) dataset becomes ONE (N*D,) vector in C order.

    Matches the existing diagnostic loader's float32 conversion; no PCA, pooling,
    feature selection, per-dataset normalization, or permutation is performed.
    """
    path = DATASET_DIR / f"n{int(num_obs)}" / split / f"{source_model}.npz"
    values = load_npz(path, keys=("x", "id", "source_model"))
    x = np.asarray(values["x"], dtype=np.float32)
    ids, sources = np.asarray(values["id"]), np.asarray(values["source_model"])
    if x.ndim != 3 or x.shape[1] != num_obs or not x.shape[0] or not x.shape[2]:
        raise ValueError(f"{path}: expected nonempty (B, {num_obs}, D), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: non-finite observations")
    if ids.shape != (len(x),) or sources.shape != (len(x),):
        raise ValueError(f"{path}: invalid id/source_model shapes")
    if ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path}: IDs must be unique integers within the file")
    if not np.all(sources == source_model):
        raise ValueError(f"{path}: source_model labels do not match the file")
    return {
        "vectors": x.reshape(len(x), -1), "id": ids, "source_model": sources,
        "num_dims": x.shape[2], "dataset_digest": _array_digest(x, ids, sources),
        "dataset_path": f"datasets/n{int(num_obs)}/{split}/{source_model}.npz",
    }


def _metadata(num_obs, num_dims):
    return {
        "schema_version": SCHEMA_VERSION, "diagnostic_space": "data",
        "representation": "flatten_C_float32", "num_obs": int(num_obs),
        "num_dims": int(num_dims), "diagnostic_dim": int(num_obs * num_dims),
        "candidate_models": np.asarray(ASSUMED_MODELS),
    }


def validate_metadata(values, *, num_obs):
    if int(values["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("Unsupported data-space schema")
    if str(values["diagnostic_space"]) != "data" or int(values["num_obs"]) != num_obs:
        raise ValueError("Diagnostic space or num_obs mismatch")
    if str(values["representation"]) != "flatten_C_float32":
        raise ValueError("Data representation mismatch")
    if int(values["diagnostic_dim"]) != num_obs * int(values["num_dims"]):
        raise ValueError("Invalid diagnostic dimension")
    if not np.array_equal(values["candidate_models"], np.asarray(ASSUMED_MODELS)):
        raise ValueError("Candidate model order mismatch")


def mmd_bandwidth2(vectors, *, max_pairs=500_000, seed=2025, chunk_size=512):
    """The existing median heuristic, without a (pairs, N*D) giant allocation."""
    vectors = np.asarray(vectors, dtype=np.float64)
    n = len(vectors)
    if n < 2 or max_pairs < 1 or chunk_size < 1:
        raise ValueError("Need >=2 reference datasets and positive pair/chunk limits")
    if n * (n - 1) // 2 <= max_pairs:
        distances = pdist(vectors, metric="sqeuclidean")
        distances = distances[distances > 0.0]
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n - 1, size=max_pairs)
        j += j >= i
        distances = np.empty(max_pairs, dtype=np.float64)
        for start in range(0, max_pairs, chunk_size):
            stop = start + chunk_size
            difference = vectors[i[start:stop]] - vectors[j[start:stop]]
            distances[start:stop] = np.einsum("ij,ij->i", difference, difference)
    return max(float(np.median(distances)), 1e-8) if distances.size else 1.0


def cross_kernel_mean(vectors, reference, bandwidth2, *, chunk_size=128):
    """Mean RBF similarity to reference per dataset; bounded 2-D work arrays."""
    x, reference = np.asarray(vectors, float), np.asarray(reference, float)
    if x.ndim != 2 or reference.ndim != 2 or x.shape[1] != reference.shape[1]:
        raise ValueError("Expected two matrices with matching feature dimensions")
    if not len(reference) or bandwidth2 <= 0 or chunk_size < 1:
        raise ValueError("Empty reference or invalid bandwidth/chunk size")
    result = np.zeros(len(x), dtype=np.float64)
    for start in range(0, len(x), chunk_size):
        block = x[start:start + chunk_size]
        total = np.zeros(len(block), dtype=np.float64)
        for ref_start in range(0, len(reference), chunk_size):
            distances = cdist(block, reference[ref_start:ref_start + chunk_size], "sqeuclidean")
            distances *= -1.0 / (2.0 * bandwidth2)
            np.exp(distances, out=distances)
            total += distances.sum(axis=1)
        result[start:start + len(block)] = total / len(reference)
    return result


def fit_non_density_reference(vectors, *, seed=2025, max_pairs=500_000):
    """Cache fitted parameters only; never cache a duplicate reference matrix."""
    vectors = np.asarray(vectors, dtype=np.float64)
    covariance = LedoitWolf().fit(vectors)
    try:
        chol = np.linalg.cholesky(covariance.covariance_)
    except np.linalg.LinAlgError as error:
        raise ValueError("Reference covariance is not positive definite") from error
    bandwidth = mmd_bandwidth2(vectors, seed=seed, max_pairs=max_pairs)
    return {
        "mean": covariance.location_, "chol": chol,
        "covariance_shrinkage": float(covariance.shrinkage_),
        "mmd_bandwidth2": bandwidth,
        "mmd_reference_kernel_mean": cross_kernel_mean(vectors, vectors, bandwidth).mean(),
    }


def non_density_scores(vectors, stats, reference_vectors):
    whitened = solve_triangular(
        stats["chol"], (np.asarray(vectors, float) - stats["mean"]).T, lower=True
    ).T
    cross = cross_kernel_mean(vectors, reference_vectors, float(stats["mmd_bandwidth2"]))
    return {
        "d_l2": np.linalg.norm(whitened, axis=1),
        "d_linf": np.max(np.abs(whitened), axis=1),
        "d_mmd": np.sqrt(np.maximum(1.0 + float(stats["mmd_reference_kernel_mean"]) - 2 * cross, 0.0)),
    }


def _density_libraries():
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    import bayesflow as bf
    import keras
    return bf, keras


def fit_density_flow(vectors, *, epochs=250, batch_size=128, depth=6,
                     widths=(256, 256, 256), learning_rate=5e-4,
                     patience=20, start_from_epoch=50, seed=2025):
    """Same flow family/training defaults as diagnostic.py; target is vec(y).

    Early stopping monitors TRAINING loss, as in the existing module. This is not
    a held-out goodness-of-fit check. Calibration data never enter this fit.
    """
    if min(epochs, batch_size, depth, patience) < 1 or learning_rate <= 0:
        raise ValueError("Invalid density training configuration")
    bf, keras = _density_libraries()
    keras.utils.set_random_seed(seed)
    adapter = bf.Adapter().convert_dtype("float64", "float32").rename("z", "inference_variables")
    dataset = bf.datasets.OfflineDataset(
        data={"z": np.asarray(vectors, dtype=np.float32)},
        batch_size=batch_size, adapter=adapter, shuffle=True,
    )
    flow = bf.approximators.ContinuousApproximator(
        inference_network=bf.networks.CouplingFlow(
            depth=depth, subnet_kwargs={"widths": tuple(widths), "activation": "mish", "norm": "layer"}
        ), standardize="inference_variables",
    )
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate, decay_steps=epochs * dataset.num_batches, alpha=1e-6
    )
    flow.compile(optimizer=keras.optimizers.Adam(schedule))
    history = flow.fit(
        dataset=dataset, epochs=epochs, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="loss", patience=patience, start_from_epoch=start_from_epoch,
            restore_best_weights=True,
        )],
    )
    return flow, {key: [float(value) for value in values] for key, values in history.history.items()}


def load_density_flow(path):
    _, keras = _density_libraries()
    return keras.saving.load_model(path, compile=False)


def density_nll(flow, vectors, *, batch_size=128):
    values = []
    for start in range(0, len(vectors), batch_size):
        log_prob = flow.log_prob({"inference_variables": np.asarray(vectors[start:start + batch_size], np.float32)})
        values.append(-np.asarray(log_prob, dtype=np.float64).reshape(-1))
    result = np.concatenate(values)
    if result.shape != (len(vectors),) or not np.isfinite(result).all():
        raise ValueError("Density flow returned invalid/non-finite log probabilities")
    return result


def load_reference(*, num_obs, model, verify_density=True):
    stats = load_npz(stats_path(num_obs=num_obs, model=model))
    validate_metadata(stats, num_obs=num_obs)
    reference = load_vectors(num_obs=num_obs, split="reference", source_model=model)
    if (str(stats["model"]) != model or
            str(stats["reference_digest"]) != reference["dataset_digest"] or
            int(stats["num_dims"]) != reference["num_dims"]):
        raise ValueError(f"{model}: reference cache is stale; refit data-space diagnostics")
    if verify_density and str(stats["density_digest"]) != file_digest(density_path(num_obs=num_obs, model=model)):
        raise ValueError(f"{model}: density checkpoint differs from the fitted reference cache")
    return stats, reference


def fit_references(*, num_obs, density_epochs=250, density_batch_size=128,
                   seed=2025, max_pairs=500_000, density_kwargs=None, overwrite=False):
    """Fit one reference+flow per assumed model, not per NPE setting or S."""
    for index, model in enumerate(ASSUMED_MODELS):
        options = dict(epochs=density_epochs, batch_size=density_batch_size, depth=6,
                       widths=(256, 256, 256), learning_rate=5e-4, patience=20,
                       start_from_epoch=50, seed=seed + index)
        options.update(density_kwargs or {})
        # Avoid accidental per-model seed loss through density_kwargs.
        if "seed" in (density_kwargs or {}):
            raise ValueError("Set the seed argument, not density_kwargs['seed']")
        config = json.dumps({"density": options, "max_pairs": max_pairs, "kernel_seed": seed + index}, sort_keys=True)
        sp, dp = stats_path(num_obs=num_obs, model=model), density_path(num_obs=num_obs, model=model)
        if not overwrite and (sp.exists() or dp.exists()):
            if not (sp.exists() and dp.exists()):
                raise RuntimeError(f"Partial reference for {model}; rerun with overwrite=True")
            stats, _ = load_reference(num_obs=num_obs, model=model)
            if str(stats["fit_config"]) != config:
                raise ValueError(f"{model}: fit settings changed; use overwrite=True")
            print(f"Reuse data-space reference: n{num_obs}/{model}")
            continue
        reference = load_vectors(num_obs=num_obs, split="reference", source_model=model)
        stats = fit_non_density_reference(reference["vectors"], seed=seed + index, max_pairs=max_pairs)
        flow, history = fit_density_flow(reference["vectors"], **options)
        dp.parent.mkdir(parents=True, exist_ok=True)
        temporary = dp.with_name(dp.stem + ".tmp.keras")
        try:
            flow.save(temporary, overwrite=True)
            temporary.replace(dp)
        finally:
            temporary.unlink(missing_ok=True)
        stats.update(_metadata(num_obs, reference["num_dims"]))
        stats.update(model=model, reference_path=reference["dataset_path"],
                     reference_digest=reference["dataset_digest"], num_reference=len(reference["id"]),
                     density_digest=file_digest(dp), fit_config=config)
        save_npz(stats, sp, overwrite=overwrite)
        dp.with_suffix(".history.json").write_text(
            json.dumps({"fit_config": json.loads(config), "history": history}, indent=2), encoding="utf-8"
        )
        del flow
        print(f"Saved data-space reference and density: n{num_obs}/{model}")


def validate_thresholds(*, num_obs, thresholds=None):
    values = load_npz(diagnostic_threshold_path(num_obs=num_obs)) if thresholds is None else thresholds
    validate_metadata(values, num_obs=num_obs)
    for index, model in enumerate(ASSUMED_MODELS):
        if str(values["reference_signatures"][index]) != file_digest(stats_path(num_obs=num_obs, model=model)):
            raise ValueError("Diagnostic thresholds are stale after a reference change")
        calibration = load_vectors(num_obs=num_obs, split="calibration", source_model=model)
        if str(values["calibration_digests"][index]) != calibration["dataset_digest"]:
            raise ValueError("Calibration datasets changed; recalibrate data-space thresholds")
    return values


def calibrate_diagnostics(*, num_obs, coverage=0.90, overwrite=False):
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie strictly between 0 and 1")
    path = diagnostic_threshold_path(num_obs=num_obs)
    if path.exists() and not overwrite:
        values = validate_thresholds(num_obs=num_obs)
        if float(values["coverage"]) != coverage:
            raise ValueError("coverage changed; recalibrate with overwrite=True")
        print(f"Reuse data-space thresholds: n{num_obs}")
        return values
    lower = {key: [] for key in DIAGNOSTICS}
    upper = {key: [] for key in DIAGNOSTICS}
    centers, signatures, digests, counts = [], [], [], []
    alpha = (1.0 - coverage) / 2.0
    num_dims = None
    for model in ASSUMED_MODELS:
        stats, reference = load_reference(num_obs=num_obs, model=model)
        calibration = load_vectors(num_obs=num_obs, split="calibration", source_model=model)
        if num_dims is None:
            num_dims = calibration["num_dims"]
        if calibration["num_dims"] != num_dims or reference["num_dims"] != num_dims:
            raise ValueError("Data dimensions differ across reference/calibration models")
        flow = load_density_flow(density_path(num_obs=num_obs, model=model))
        nll = density_nll(flow, calibration["vectors"])
        scores = non_density_scores(calibration["vectors"], stats, reference["vectors"])
        center = float(nll.mean())
        scores["d_density"] = nll - center
        for key in DIAGNOSTICS:
            low, high = np.quantile(scores[f"d_{key}"], [alpha, 1 - alpha])
            if not np.isfinite([low, high]).all() or high <= 0:
                raise ValueError(f"{model}/{key}: invalid upper bound for rho=d/upper")
            lower[key].append(low)
            upper[key].append(high)
        centers.append(center)
        signatures.append(file_digest(stats_path(num_obs=num_obs, model=model)))
        digests.append(calibration["dataset_digest"])
        counts.append(len(calibration["id"]))
        del flow
    values = _metadata(num_obs, num_dims)
    values.update(coverage=coverage, density_center=np.asarray(centers),
                  reference_signatures=np.asarray(signatures), calibration_digests=np.asarray(digests),
                  num_calibration=np.asarray(counts))
    for key in DIAGNOSTICS:
        values[f"d_{key}_lower"] = np.asarray(lower[key])
        values[f"d_{key}_upper"] = np.asarray(upper[key])
    save_npz(values, path, overwrite=overwrite)
    print(f"Saved data-space thresholds: {path}")
    return values


def run_diagnostics(*, num_obs, split="simulated", sources=None, overwrite=False):
    """Save (B,4) scores; source model and candidate model remain distinct axes."""
    sources = tuple(SOURCE_MODELS if sources is None and split == "simulated" else
                    ASSUMED_MODELS if sources is None else sources)
    thresholds = validate_thresholds(num_obs=num_obs)
    signature = file_digest(diagnostic_threshold_path(num_obs=num_obs))
    datasets, outputs = {}, {}
    for source in sources:
        dataset = load_vectors(num_obs=num_obs, split=split, source_model=source)
        if dataset["num_dims"] != int(thresholds["num_dims"]):
            raise ValueError(f"{source}: data dimension differs from the diagnostic reference")
        path = diagnostic_result_path(num_obs=num_obs, split=split, source_model=source)
        if path.exists() and not overwrite:
            previous = load_npz(path, keys=("dataset_digest", "thresholds_digest"))
            if (str(previous["dataset_digest"]) != dataset["dataset_digest"] or
                    str(previous["thresholds_digest"]) != signature):
                raise ValueError(f"{path}: stale diagnostic scores; rerun with overwrite=True")
            print(f"Reuse data-space scores: {path}")
            continue
        datasets[source] = dataset
        values = _metadata(num_obs, dataset["num_dims"])
        values.update(id=dataset["id"], source_model=dataset["source_model"], split=split,
                      dataset_digest=dataset["dataset_digest"], thresholds_digest=signature)
        values.update({f"d_{key}": np.empty((len(dataset["id"]), len(ASSUMED_MODELS))) for key in DIAGNOSTICS})
        outputs[source] = values
    if not datasets:
        return
    # Load one candidate density at a time; score every source with the same model.
    for index, model in enumerate(ASSUMED_MODELS):
        stats, reference = load_reference(num_obs=num_obs, model=model)
        flow = load_density_flow(density_path(num_obs=num_obs, model=model))
        for source, dataset in datasets.items():
            scores = non_density_scores(dataset["vectors"], stats, reference["vectors"])
            scores["d_density"] = density_nll(flow, dataset["vectors"]) - thresholds["density_center"][index]
            for key in DIAGNOSTICS:
                outputs[source][f"d_{key}"][:, index] = scores[f"d_{key}"]
        del flow
    for source, values in outputs.items():
        for key in DIAGNOSTICS:
            score = values[f"d_{key}"]
            low, high = thresholds[f"d_{key}_lower"][None, :], thresholds[f"d_{key}_upper"][None, :]
            if not np.isfinite(score).all():
                raise ValueError(f"{source}/{key}: non-finite scores")
            values[f"rho_{key}"] = score / high
            values[f"typical_{key}"] = (score >= low) & (score <= high)
        path = diagnostic_result_path(num_obs=num_obs, split=split, source_model=source)
        save_npz(values, path, overwrite=overwrite)
        print(f"Saved data-space scores: {path}")

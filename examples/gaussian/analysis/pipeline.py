"""Run the complete Gaussian analysis pipeline."""
# fmt: off

from pathlib import Path

from ..analytic.gold import GOLD_DIR, compute_gold_file
from ..approximators.config import DIRECT_SCORING_RULES
from ..config import ASSUMED_MODELS, SOURCE_MODELS
from .comparison import comparison_paths, run_direct_comparison, run_indirect_comparison
from .diagnostic import (
    diagnostic_result_path,
    diagnostic_threshold_path,
    direct_density_path,
    direct_reference_path,
    fit_direct_reference,
    fit_indirect_references,
    indirect_density_path,
    indirect_reference_path,
    calibrate_direct_diagnostics,
    calibrate_indirect_diagnostics,
    run_direct_diagnostics,
    run_indirect_diagnostics,
)
from .inference import inference_path, run_direct_file, run_indirect_file
from .metric import metrics_path, run_direct_metrics_file, run_indirect_metrics_file
from .threshold import run_direct_thresholds, run_indirect_thresholds, threshold_path

# ---------------------------------------------------------------------
# Global experiment settings
# ---------------------------------------------------------------------

NUM_OBS_VALUES = (10, 100)

INDIRECT_SUMMARY_DIMS = (20, 40, 80)

DIRECT_SUMMARY_DIM = 12

SCORING_RULES = DIRECT_SCORING_RULES

NUM_POSTERIOR_SAMPLES = 2048

NUM_MMD_SAMPLES = 1024

SEED = 2025

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def needs_run(path, *, overwrite=False):
    return overwrite or not Path(path).exists()

def gold_file(*, split, source_model, num_obs):
    return GOLD_DIR / f"n{num_obs}" / split / f"{source_model}.npz"

# ---------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------

def run_gold(*, num_obs, overwrite=False):
    """
    Compute analytical gold results required by:
        benchmark thresholds
        simulated evaluation
    """
    split_sources = {"benchmark": ASSUMED_MODELS, "simulated": SOURCE_MODELS}
    for split, sources in split_sources.items():
        for source_model in sources:
            path = gold_file(split=split, source_model=source_model, num_obs=num_obs)
            if not needs_run(path, overwrite=overwrite):
                print(f"Skip existing: {path}")
                continue
            compute_gold_file(
                split=split,
                source_model=source_model,
                num_obs=num_obs,
                num_samples=NUM_POSTERIOR_SAMPLES,
                seed=SEED,
                overwrite=overwrite,
            )

# ---------------------------------------------------------------------
# Indirect
# ---------------------------------------------------------------------

def run_indirect_inference_and_metrics(*, split, sources, num_obs, summary_dim, overwrite=False, variant="baseline"):
    for source_model in sources:
        inf_path = inference_path(
            method="indirect", split=split, source_model=source_model, num_obs=num_obs, summary_dim=summary_dim,
            variant=variant
        )
        if needs_run(inf_path, overwrite=overwrite):
            run_indirect_file(
                split=split,
                source_model=source_model,
                num_obs=num_obs,
                summary_dim=summary_dim,
                num_samples=NUM_POSTERIOR_SAMPLES,
                seed=SEED,
                overwrite=overwrite,
                variant=variant,
            )
        else:
            print(f"Skip existing: {inf_path}")
        metric_path = metrics_path(
            method="indirect", split=split, source_model=source_model, num_obs=num_obs, summary_dim=summary_dim,
            variant=variant
        )
        if needs_run(metric_path, overwrite=overwrite):
            run_indirect_metrics_file(
                split=split,
                source_model=source_model,
                num_obs=num_obs,
                summary_dim=summary_dim,
                num_mmd_samples=NUM_MMD_SAMPLES,
                overwrite=overwrite,
                variant=variant,
            )
        else:
            print(f"Skip existing: {metric_path}")

def run_indirect_reference(*, num_obs, summary_dim, overwrite=False, variant="baseline"):
    reference_files = []
    for model in ASSUMED_MODELS:
        reference_files.extend(
            [
                indirect_reference_path(num_obs=num_obs, summary_dim=summary_dim, model=model, variant=variant),
                indirect_density_path(num_obs=num_obs, summary_dim=summary_dim, model=model, variant=variant),
            ]
        )
    all_exist = all(path.exists() for path in reference_files)
    none_exist = not any(path.exists() for path in reference_files)
    if all_exist and not overwrite:
        print("Skip existing indirect diagnostic references")
        return
    if not none_exist and not all_exist and not overwrite:
        raise RuntimeError("Indirect diagnostic reference files are only partially present. Rerun with overwrite=True.")
    fit_indirect_references(num_obs=num_obs, summary_dim=summary_dim, seed=SEED, overwrite=overwrite, variant=variant)

def run_indirect_analysis(*, num_obs, summary_dim, overwrite=False, variant="baseline"):
    print(f"\nINDIRECT: N={num_obs}, S={summary_dim}\n")
    # --------------------------------------------------
    # Benchmark → inference error thresholds
    # --------------------------------------------------
    run_indirect_inference_and_metrics(
        split="benchmark", sources=ASSUMED_MODELS, num_obs=num_obs, summary_dim=summary_dim, overwrite=overwrite,
        variant=variant
    )
    error_threshold_path = threshold_path(method="indirect", num_obs=num_obs, summary_dim=summary_dim, variant=variant)
    if needs_run(error_threshold_path, overwrite=overwrite):
        run_indirect_thresholds(num_obs=num_obs, summary_dim=summary_dim, overwrite=overwrite, variant=variant)
    else:
        print(f"Skip existing: {error_threshold_path}")
    # --------------------------------------------------
    # Reference + calibration → diagnostic thresholds
    # --------------------------------------------------
    run_indirect_reference(num_obs=num_obs, summary_dim=summary_dim, overwrite=overwrite, variant=variant)
    diag_threshold_path = diagnostic_threshold_path(method="indirect", num_obs=num_obs, summary_dim=summary_dim, variant=variant)
    if needs_run(diag_threshold_path, overwrite=overwrite):
        calibrate_indirect_diagnostics(num_obs=num_obs, summary_dim=summary_dim, overwrite=overwrite, variant=variant)
    else:
        print(f"Skip existing: {diag_threshold_path}")
    # --------------------------------------------------
    # Simulated M1-M12 → actual inference errors
    # --------------------------------------------------
    run_indirect_inference_and_metrics(
        split="simulated", sources=SOURCE_MODELS, num_obs=num_obs, summary_dim=summary_dim, overwrite=overwrite,
        variant=variant
    )
    # --------------------------------------------------
    # Simulated M1-M12 → diagnostics
    # --------------------------------------------------
    missing_sources = []
    for source in SOURCE_MODELS:
        path = diagnostic_result_path(
            method="indirect", split="simulated", source_model=source, num_obs=num_obs, summary_dim=summary_dim,
            variant=variant
        )
        if needs_run(path, overwrite=overwrite):
            missing_sources.append(source)
    if missing_sources:
        run_indirect_diagnostics(
            split="simulated", num_obs=num_obs, summary_dim=summary_dim, sources=missing_sources, overwrite=overwrite,
            variant=variant
        )
    # --------------------------------------------------
    # Final comparison
    # --------------------------------------------------
    paths = comparison_paths(method="indirect", num_obs=num_obs, summary_dim=summary_dim, variant=variant)
    comparison_complete = all(path.exists() for path in paths.values())
    if overwrite or not comparison_complete:
        run_indirect_comparison(num_obs=num_obs, summary_dim=summary_dim, overwrite=True, variant=variant)
    else:
        print(f"Skip existing comparison: {paths['data'].parent}")

# ---------------------------------------------------------------------
# Direct
# ---------------------------------------------------------------------

def run_direct_inference_and_metrics(*, split, sources, num_obs, summary_dim, scoring_rule, overwrite=False):
    for source_model in sources:
        inf_path = inference_path(
            method="direct",
            split=split,
            source_model=source_model,
            num_obs=num_obs,
            summary_dim=summary_dim,
            scoring_rule=scoring_rule,
        )
        if needs_run(inf_path, overwrite=overwrite):
            run_direct_file(
                split=split,
                source_model=source_model,
                num_obs=num_obs,
                summary_dim=summary_dim,
                scoring_rule=scoring_rule,
                overwrite=overwrite,
            )
        else:
            print(f"Skip existing: {inf_path}")
        metric_path = metrics_path(
            method="direct",
            split=split,
            source_model=source_model,
            num_obs=num_obs,
            summary_dim=summary_dim,
            scoring_rule=scoring_rule,
        )
        if needs_run(metric_path, overwrite=overwrite):
            run_direct_metrics_file(
                split=split,
                source_model=source_model,
                num_obs=num_obs,
                summary_dim=summary_dim,
                scoring_rule=scoring_rule,
                overwrite=overwrite,
            )
        else:
            print(f"Skip existing: {metric_path}")

def run_direct_reference(*, num_obs, summary_dim, scoring_rule, overwrite=False):
    reference_path = direct_reference_path(num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
    density_path = direct_density_path(num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
    both_exist = reference_path.exists() and density_path.exists()
    neither_exists = not reference_path.exists() and not density_path.exists()
    if both_exist and not overwrite:
        print("Skip existing direct diagnostic reference")
        return
    if not neither_exists and not both_exist and not overwrite:
        raise RuntimeError("Direct diagnostic reference is only partially present. Rerun with overwrite=True.")
    fit_direct_reference(
        num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule, seed=SEED, overwrite=overwrite
    )

def run_direct_analysis(*, num_obs, summary_dim, scoring_rule, overwrite=False):
    print(f"\nDIRECT: {scoring_rule}, N={num_obs}, S={summary_dim}\n")
    # --------------------------------------------------
    # Benchmark → direct PMP error thresholds
    # --------------------------------------------------
    run_direct_inference_and_metrics(
        split="benchmark",
        sources=ASSUMED_MODELS,
        num_obs=num_obs,
        summary_dim=summary_dim,
        scoring_rule=scoring_rule,
        overwrite=overwrite,
    )
    error_threshold_path = threshold_path(
        method="direct", num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule
    )
    if needs_run(error_threshold_path, overwrite=overwrite):
        run_direct_thresholds(num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule, overwrite=overwrite)
    else:
        print(f"Skip existing: {error_threshold_path}")
    # --------------------------------------------------
    # Mixed reference → diagnostic reference
    # --------------------------------------------------
    run_direct_reference(num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule, overwrite=overwrite)
    diag_threshold_path = diagnostic_threshold_path(
        method="direct", num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule
    )
    if needs_run(diag_threshold_path, overwrite=overwrite):
        calibrate_direct_diagnostics(
            num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule, overwrite=overwrite
        )
    else:
        print(f"Skip existing: {diag_threshold_path}")
    # --------------------------------------------------
    # Simulated M1-M12 inference
    # --------------------------------------------------
    run_direct_inference_and_metrics(
        split="simulated",
        sources=SOURCE_MODELS,
        num_obs=num_obs,
        summary_dim=summary_dim,
        scoring_rule=scoring_rule,
        overwrite=overwrite,
    )
    # --------------------------------------------------
    # Simulated M1-M12 diagnostics
    # --------------------------------------------------
    missing_sources = []
    for source in SOURCE_MODELS:
        path = diagnostic_result_path(
            method="direct",
            split="simulated",
            source_model=source,
            num_obs=num_obs,
            summary_dim=summary_dim,
            scoring_rule=scoring_rule,
        )
        if needs_run(path, overwrite=overwrite):
            missing_sources.append(source)
    if missing_sources:
        run_direct_diagnostics(
            split="simulated",
            num_obs=num_obs,
            summary_dim=summary_dim,
            scoring_rule=scoring_rule,
            sources=missing_sources,
            overwrite=overwrite,
        )
    # --------------------------------------------------
    # Final comparison
    # --------------------------------------------------
    paths = comparison_paths(method="direct", num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule)
    comparison_complete = all(path.exists() for path in paths.values())
    if overwrite or not comparison_complete:
        run_direct_comparison(num_obs=num_obs, summary_dim=summary_dim, scoring_rule=scoring_rule, overwrite=True)
    else:
        print(f"Skip existing comparison: {paths['data'].parent}")

# ---------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------

def run_all(*, overwrite=False):
    # Gold is shared by all methods.
    for num_obs in NUM_OBS_VALUES:
        run_gold(num_obs=num_obs, overwrite=overwrite)
    # Indirect
    for num_obs in NUM_OBS_VALUES:
        for summary_dim in INDIRECT_SUMMARY_DIMS:
            run_indirect_analysis(num_obs=num_obs, summary_dim=summary_dim, overwrite=overwrite)
    # Direct
    for num_obs in NUM_OBS_VALUES:
        for scoring_rule in SCORING_RULES:
            run_direct_analysis(
                num_obs=num_obs, summary_dim=DIRECT_SUMMARY_DIM, scoring_rule=scoring_rule, overwrite=overwrite
            )

if __name__ == "__main__":
    run_all(overwrite=False)

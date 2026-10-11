"""Run all diffusion inference jobs."""

from ..approximators.config import DIRECT_SCORING_RULES, SUMMARY_MULTIPLIERS
from ..config import ASSUMED_MODELS, SIMULATED_DATASETS, normalize_variant
from .inference import inference_path, run_direct, run_indirect

NUM_SAMPLES = 2048
SEED = 2025
BATCH_SIZE = 8

def run_all_indirect(*, variant="baseline", summary_multipliers=SUMMARY_MULTIPLIERS, overwrite=False):
    """Run indirect inference for benchmark and simulated datasets."""
    variant = normalize_variant(variant)
    split_sources = {"benchmark": ASSUMED_MODELS, "simulated": SIMULATED_DATASETS}
    for summary_multiplier in summary_multipliers:
        for split, sources in split_sources.items():
            for source in sources:
                path = inference_path(
                    method="indirect", split=split, source=source, summary_multiplier=summary_multiplier, variant=variant
                )
                if path.exists() and not overwrite:
                    print(f"Skip existing: {path}")
                    continue
                run_indirect(
                    split=split,
                    source=source,
                    summary_multiplier=summary_multiplier,
                    variant=variant,
                    num_samples=NUM_SAMPLES,
                    seed=SEED,
                    batch_size=BATCH_SIZE,
                    overwrite=overwrite,
                )

def run_all_direct(*, overwrite=False):
    """Run direct inference for benchmark and simulated datasets."""
    split_sources = {"benchmark": ASSUMED_MODELS, "simulated": SIMULATED_DATASETS}
    for scoring_rule in DIRECT_SCORING_RULES:
        for split, sources in split_sources.items():
            for source in sources:
                path = inference_path(method="direct", split=split, source=source, scoring_rule=scoring_rule)
                if path.exists() and not overwrite:
                    print(f"Skip existing: {path}")
                    continue
                run_direct(split=split, source=source, scoring_rule=scoring_rule, overwrite=overwrite)


def run_empirical(*, batch_size=64, overwrite=False, variant="baseline", summary_multipliers=SUMMARY_MULTIPLIERS):
    """Run empirical inference; non-baseline variants include only indirect estimators."""
    variant = normalize_variant(variant)
    for summary_multiplier in summary_multipliers:
        path = inference_path(
            method="indirect", split="empirical", source="empirical", summary_multiplier=summary_multiplier, variant=variant
        )
        if not path.exists() or overwrite:
            run_indirect(
                split="empirical", source="empirical", summary_multiplier=summary_multiplier,
                num_samples=NUM_SAMPLES, seed=SEED, batch_size=batch_size, overwrite=overwrite,
                variant=variant,
            )
    if variant != "baseline":
        return
    for scoring_rule in DIRECT_SCORING_RULES:
        path = inference_path(method="direct", split="empirical", source="empirical", scoring_rule=scoring_rule)
        if not path.exists() or overwrite:
            run_direct(split="empirical", source="empirical", scoring_rule=scoring_rule, overwrite=overwrite)

if __name__ == "__main__":
    run_all_indirect()
    run_all_direct()

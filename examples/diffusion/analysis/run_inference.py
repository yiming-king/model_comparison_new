"""Run all diffusion inference jobs."""

from ..approximators.config import DIRECT_SCORING_RULES, SUMMARY_MULTIPLIERS
from ..config import ASSUMED_MODELS, SIMULATED_DATASETS
from .inference import inference_path, run_direct, run_indirect

NUM_SAMPLES = 2048
SEED = 2025
BATCH_SIZE = 8

def run_all_indirect(*, overwrite=False):
    """Run indirect inference for benchmark and simulated datasets."""
    split_sources = {"benchmark": ASSUMED_MODELS, "simulated": SIMULATED_DATASETS}
    for summary_multiplier in SUMMARY_MULTIPLIERS:
        for split, sources in split_sources.items():
            for source in sources:
                path = inference_path(
                    method="indirect", split=split, source=source, summary_multiplier=summary_multiplier
                )
                if path.exists() and not overwrite:
                    print(f"Skip existing: {path}")
                    continue
                run_indirect(
                    split=split,
                    source=source,
                    summary_multiplier=summary_multiplier,
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

if __name__ == "__main__":
    run_all_indirect()
    run_all_direct()

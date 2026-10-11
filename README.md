# model_comparison_new

## RDM indirect standardization comparison

The existing RDM pipeline accepts `variant="baseline"` (also `"default"`) and
`variant="inference_only"`. The latter uses
`standardize="inference_variables"`; the baseline uses `standardize="all"`.
The variant settings are in `examples/diffusion/config.py`.
`TrainingConfig.standardize` and checkpoint paths are in
`examples/diffusion/approximators/config.py`.

The adapter still combines `rt` and `conditions` into `summary_variables` and
`alpha`, `nu`, and `tau` into `inference_variables`. This experiment changes
automatic standardization of the **summary network input**. It does not
directly standardize or unstandardize the learned output `h(y)`.

All other training settings stay the same: DeepSet, CouplingFlow, cosine decay,
256 epochs, 128 batches per epoch, batch size 64, learning rate `1e-4`, seed
2025, and `summary_base_distribution=None`. Use the same dependency environment
as for the baseline; changing library versions would introduce another factor.

| Output | baseline / default | inference_only |
| --- | --- | --- |
| Posterior networks and histories | `examples/diffusion/networks/` | `examples/diffusion/networks/ablation/inference_only/` |
| Inference, metrics, thresholds, diagnostics, comparison, plots | `examples/diffusion/results/` | `examples/diffusion/results/ablation/inference_only/` |

Each root retains the existing subdirectory and filename conventions. The
diagnostic density models and summary references are also variant-specific.
Raw reference, calibration, benchmark, simulated, and empirical datasets and
Stan gold remain at their existing shared paths. Do not regenerate them for
the new variant.

### Train the new networks

Run from the repository root in the existing research environment:

```bash
python -m examples.diffusion.approximators.indirect \
  --variant inference_only --summary-multipliers 1 2 4
```

This trains four candidate models at each selected summary multiplier.
To restrict the experiment to `S=4D`, use `--summary-multipliers 4`, and set
`MULTIPLIERS = (4,)` in the analysis block below. For `S=4D`, the model-specific
summary dimensions are 16, 20, 20, and 28 for m0, m1, m2, and m3.
Existing checkpoints are skipped unless `--overwrite` is supplied.

The original command without arguments still trains the baseline.
For individual training calls, pass both a matching configuration and variant:

```python
from examples.diffusion.approximators.config import TrainingConfig
from examples.diffusion.approximators.indirect import train_one

train_one(
    "m3",
    TrainingConfig(summary_dim=28, standardize="inference_variables"),
    variant="inference_only",
)
```

A mismatched configuration and variant raises an error before training or
saving, so an inference-only configuration cannot silently replace a baseline
checkpoint.

### Compute results using the shared evaluation data

The block below fills missing results for both variants, including empirical
data. Complete existing results are reused. It uses the existing functions;
no additional pipeline script is required. Both variants' selected posterior
networks, raw datasets, and gold must already be available locally.

```bash
python - <<'PY'
from examples.diffusion.config import ASSUMED_MODELS, SIMULATED_DATASETS
from examples.diffusion.analysis.inference import inference_path, run_indirect
from examples.diffusion.analysis.metric import metrics_path, run_indirect_metrics
from examples.diffusion.analysis.threshold import threshold_path, run_indirect_thresholds
from examples.diffusion.analysis.diagnostic import (
    fit_indirect_references, calibrate_indirect, run_indirect_diagnostics,
)
from examples.diffusion.analysis.comparison import comparison_paths, run_indirect_comparison

VARIANTS = ("baseline", "inference_only")
MULTIPLIERS = (1, 2, 4)
SOURCES = {
    "benchmark": ASSUMED_MODELS,
    "simulated": SIMULATED_DATASETS,
    "empirical": ("empirical",),
}

for variant in VARIANTS:
    for multiplier in MULTIPLIERS:
        options = dict(summary_multiplier=multiplier, variant=variant)
        for split, sources in SOURCES.items():
            for source in sources:
                job = dict(split=split, source=source, **options)
                if not inference_path(method="indirect", **job).exists():
                    run_indirect(**job, num_samples=2048, seed=2025, batch_size=8)
                if not metrics_path(method="indirect", **job).exists():
                    run_indirect_metrics(**job, num_mmd_samples=1024, seed=2025)

        if not threshold_path(method="indirect", **options).exists():
            run_indirect_thresholds(
                **options, posterior_quantile=0.95, signed_error_coverage=0.90,
            )

        fit_indirect_references(**options, density_epochs=250, seed=2025)
        calibrate_indirect(**options, coverage=0.90)
        for split in ("simulated", "empirical"):
            for source in SOURCES[split]:
                run_indirect_diagnostics(split=split, source=source, **options)

        paths = comparison_paths(method="indirect", **options)
        if not all(path.exists() for path in paths.values()):
            # Rebuild an incomplete CSV set from the saved numeric inputs.
            run_indirect_comparison(
                **options, splits=("simulated", "empirical"), overwrite=True,
            )
PY
```

Inference-error thresholds (posterior MMD, logML, PMP) and summary-space
diagnostic thresholds are calibrated separately for each variant. Shared raw
data do not imply shared learned summaries, density models, or thresholds.

If a checkpoint is deliberately retrained at the same path, regenerate its
derived inference, metrics, references/density flows, both sets of thresholds,
diagnostics, and comparisons with `overwrite=True`. The existence checks above
do not detect whether an existing result has become stale.

### Read and plot the comparison

In the existing RDM notebook, select the variant when loading:

```python
import matplotlib.pyplot as plt
from examples.diffusion.analysis.plot import load_runs, plot_metric_grid

for variant in ("baseline", "inference_only"):
    runs = load_runs("indirect", summary_multipliers=(4,), variant=variant)
    print(variant)
    fig, axes = plot_metric_grid(runs, metric="posterior_mmd", dataset_type="both")
    plt.show()
    plt.close(fig)
```

Use `metric="logml"` or `metric="pmp"` for the other existing grids.
Load each variant separately so that the existing plot layout remains usable.
The comparison CSVs now include a `variant` column; old baseline CSVs acquire
the label when read through `load_comparison`. Detection summaries keep the
variants separate if their tables are concatenated.

To save the existing grids under the selected variant's result root:

```bash
python -m examples.diffusion.analysis.plot \
  --method indirect --variant inference_only --summary-multipliers 1 2 4
```

Direct networks remain baseline-only. Calls without a variant preserve the
original baseline paths and behavior. The `run_inference` module's batch
runners accept variant as a Python keyword; use the analysis block above,
not a `--variant` argument to that module's existing command-line entry point.

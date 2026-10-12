"""Independent Gaussian data-space diagnostic and CSV-merge pipeline."""
import argparse
from itertools import product

from .diagnostic_dataspace import fit_references, calibrate_diagnostics, run_diagnostics
from .comparison_dataspace import comparison_path, run_comparison


def run(*, num_obs=(10,), variants=("inference_only",), summary_dims=(40, 80),
        stage="all", force_diagnostics=False):
    if stage not in ("all", "diagnostics", "comparison"):
        raise ValueError(stage)
    # Before fitting the four density flows, check selected old CSV inputs.
    if stage in ("all", "comparison"):
        missing = [str(comparison_path(num_obs=n, summary_dim=s, variant=v)[0])
                   for n, v, s in product(num_obs, variants, summary_dims)
                   if not comparison_path(num_obs=n, summary_dim=s, variant=v)[0].is_file()]
        if missing:
            raise FileNotFoundError("Missing original comparison CSVs:\n" + "\n".join(missing))

    for n in num_obs:
        if stage in ("all", "diagnostics"):
            fit_references(num_obs=n, overwrite=force_diagnostics)
            calibrate_diagnostics(num_obs=n, overwrite=force_diagnostics)
            run_diagnostics(num_obs=n, overwrite=force_diagnostics)
        if stage in ("all", "comparison"):
            for v, s in product(variants, summary_dims):
                run_comparison(num_obs=n, summary_dim=s, variant=v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-obs", type=int, nargs="+", default=[10])
    parser.add_argument("--variants", nargs="+", default=["inference_only"])
    parser.add_argument("--summary-dims", type=int, nargs="+", default=[40, 80])
    parser.add_argument("--stage", choices=("all", "diagnostics", "comparison"), default="all")
    parser.add_argument("--force-diagnostics", action="store_true")
    args = parser.parse_args()
    run(num_obs=args.num_obs, variants=args.variants, summary_dims=args.summary_dims,
        stage=args.stage, force_diagnostics=args.force_diagnostics)

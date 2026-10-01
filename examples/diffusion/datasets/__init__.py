"""Read fixed diffusion datasets."""

from importlib import import_module

from .datasets import load_benchmark_dataset, load_json_directory, load_simulated_dataset

__all__ = (
    "load_json_directory",
    "load_simulated_dataset",
    "load_benchmark_dataset",
    "load_diagnostic_dataset",
    "dataset_path",
    "generate_all",
)


def __getattr__(name):
    # Fixed JSON datasets do not need the simulator or a Keras backend.
    if name in {"load_diagnostic_dataset", "dataset_path", "generate_all"}:
        return getattr(import_module(".diagnostic", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

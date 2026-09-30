from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import bayesflow as bf
import keras
import numpy as np

from ..config import (
    MODELS,
    SUMMARY_MULTIPLIERS,
    TrainingConfig,
    ensure_dirs,
    get_history_path,
    get_name,
    get_path,
)
from ..networks import PosteriorNetwork, SummaryNetwork
from ..simulators import SIMULATORS


models = list(MODELS)


class SavedHistory:
    def __init__(self, history: dict[str, list[float]]):
        self.history = history


def build_adapter() -> bf.adapters.Adapter:
    return (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .as_set(["rt", "conditions"])
        .concatenate(
            ["rt", "conditions"], into="summary_variables"
        )  # (batch_size, n_trials, 2)
        .concatenate(
            ["alpha", "nu", "tau"], into="inference_variables"
        )  # (batch_size, num_parameters)
    )


def get_simulator(model: str):
    return SIMULATORS[model]


def generate_validation_data(
    model: str,
    validation_size: int,
    validation_seed: int,
) -> dict[str, np.ndarray]:
    """Generate a reproducible validation set without changing training RNG state."""
    if validation_size < 1:
        raise ValueError("validation_size must be at least 1")

    rng_state = np.random.get_state()
    try:
        np.random.seed(validation_seed + MODELS.index(model))
        return get_simulator(model).sample(validation_size)
    finally:
        np.random.set_state(rng_state)


def build_workflow(
    model: str, config: TrainingConfig = TrainingConfig()
) -> bf.BasicWorkflow:
    return bf.BasicWorkflow(
        simulator=get_simulator(model),
        adapter=build_adapter(),
        inference_network=PosteriorNetwork(),
        summary_network=SummaryNetwork(
            summary_dim=config.summary_dim_for(model),
            embed_dim=config.embed_dim,
            base_distribution=config.summary_base_distribution,
        ),
        standardize="all",
    )


def load_approximator(
    model: str, config: TrainingConfig = TrainingConfig(), approximation: str = "NPE"
):
    name = get_name(model, approximation, config.summary_label)
    path = get_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Trained approximator not found: {path}")
    # Preserve historical DeepSet attention computations when loading old archives.
    from ...gaussian.approximators.legacy_npe import load_checkpoint

    return load_checkpoint(path)


def save_history(
    history,
    model: str,
    config: TrainingConfig = TrainingConfig(),
    approximation: str = "NPE",
):
    name = get_name(model, approximation, config.summary_label)
    path = get_history_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    with path.open("w") as f:
        json.dump(serializable, f, indent=2)
    return path


def load_history(
    model: str, config: TrainingConfig = TrainingConfig(), approximation: str = "NPE"
):
    name = get_name(model, approximation, config.summary_label)
    path = get_history_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Training history not found: {path}")
    with path.open("r") as f:
        return SavedHistory(json.load(f))


def train_approximator(
    model: str,
    config: TrainingConfig = TrainingConfig(),
    overwrite: bool = False,
    validation_size: int | None = None,
    validation_freq: int = 1,
    validation_seed: int = 2026,
):
    ensure_dirs()
    name = get_name(model, "NPE", config.summary_label)
    path = get_path(name)
    if path.exists() and not overwrite:
        history = (
            load_history(model, config=config)
            if get_history_path(name).exists()
            else None
        )
        return load_approximator(model, config=config), history

    workflow = build_workflow(model, config=config)

    fit_kwargs = {}
    if validation_size is not None:
        if validation_freq < 1:
            raise ValueError("validation_freq must be at least 1")
        fit_kwargs["validation_data"] = generate_validation_data(
            model,
            validation_size,
            validation_seed,
        )
        fit_kwargs["validation_freq"] = validation_freq

    history = workflow.fit_online(
        epochs=config.epochs,
        batch_size=config.batch_size,
        num_batches_per_epoch=config.num_batches,
        **fit_kwargs,
    )
    workflow.approximator.save(path)
    save_history(history, model, config=config)
    return workflow.approximator, history


def train_approximators(
    config: TrainingConfig = TrainingConfig(),
    overwrite: bool = False,
    validation_size: int | None = None,
    validation_freq: int = 1,
    validation_seed: int = 2026,
):
    return {
        model: train_approximator(
            model,
            config=config,
            overwrite=overwrite,
            validation_size=validation_size,
            validation_freq=validation_freq,
            validation_seed=validation_seed,
        )
        for model in models
    }


def get_approximators(load: bool = True, config: TrainingConfig = TrainingConfig()):
    output = {}
    for model in models:
        if load:
            approximator = load_approximator(model, config=config)
            workflow = None
        else:
            workflow = build_workflow(model, config=config)
            approximator = workflow.approximator
        output[model] = {
            "workflow": workflow,
            "approximator": approximator,
            "config": config,
        }
    return output


if __name__ == "__main__":
    default_config = TrainingConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=default_config.epochs)
    parser.add_argument("--batch-size", type=int, default=default_config.batch_size)
    parser.add_argument("--num-batches", type=int, default=default_config.num_batches)
    parser.add_argument("--summary-dim", type=int, default=default_config.summary_dim)
    parser.add_argument(
        "--summary-multiplier", type=int, choices=SUMMARY_MULTIPLIERS,
        default=default_config.summary_multiplier
    )
    parser.add_argument("--embed-dim", type=int, default=default_config.embed_dim)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-summary-mmd", action="store_true")
    parser.add_argument("--run-suffix", type=str, default=None)
    parser.add_argument(
        "--validation-size",
        type=int,
        default=None,
        help="Number of fixed validation datasets generated before training.",
    )
    parser.add_argument(
        "--validation-freq",
        type=int,
        default=1,
        help="Run validation every N epochs (default: 1).",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=2026,
        help="Seed used to generate reproducible model-specific validation data.",
    )
    args = parser.parse_args()

    if args.validation_size is not None and args.validation_size < 1:
        parser.error("--validation-size must be at least 1")
    if args.validation_freq < 1:
        parser.error("--validation-freq must be at least 1")
    if args.embed_dim < 1:
        parser.error("--embed-dim must be at least 1")

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        summary_dim=args.summary_dim,
        summary_multiplier=args.summary_multiplier,
        embed_dim=args.embed_dim,
        summary_base_distribution=(None if args.no_summary_mmd else "normal"),
        run_suffix=args.run_suffix,
    )
    train_approximators(
        config=config,
        overwrite=args.overwrite,
        validation_size=args.validation_size,
        validation_freq=args.validation_freq,
        validation_seed=args.validation_seed,
    )


# cd /Users/yimingzang/Documents/Project/benchmark2

# KERAS_BACKEND=tensorflow \
# MPLCONFIGDIR=/private/tmp/matplotlib \
# /opt/anaconda3/envs/benchmark2/bin/python \
# -m benchmark.examples.diffusion.approximators.indirect \
# --summary-multiplier 4 \
# --epochs 100 \
# --batch-size 64 \
# --num-batches 128 \
# --no-summary-mmd \
# --run-suffix noMMD

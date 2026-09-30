"""Train indirect NPE models for the diffusion case study."""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import itertools
import json
from dataclasses import asdict

import bayesflow as bf
import keras

from ..config import ASSUMED_MODELS, PARAM_DIMS
from ..simulators import SIMULATORS
from .config import SUMMARY_MULTIPLIERS, TrainingConfig, checkpoint_path, history_path

MODELS = ASSUMED_MODELS

def build_adapter():
    return (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .as_set(["rt", "conditions"])
        .concatenate(["rt", "conditions"], into="summary_variables")
        .concatenate(["alpha", "nu", "tau"], into="inference_variables")
    )

def build_workflow(model: str, config: TrainingConfig):
    keras.utils.set_random_seed(config.seed)
    summary_network = bf.networks.DeepSet(summary_dim=config.summary_dim, base_distribution=config.summary_base_distribution,)
    inference_network = bf.networks.CouplingFlow()

    learning_rate = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.learning_rate, decay_steps=config.epochs * config.num_batches
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    workflow = bf.BasicWorkflow(
        simulator=SIMULATORS[model],
        adapter=build_adapter(),
        summary_network=summary_network,
        inference_network=inference_network,
        standardize="all",
        optimizer=optimizer,
    )

    return workflow

def train_one(model: str, config: TrainingConfig, *, save=True, overwrite=False):
    network_path = checkpoint_path(config, model=model)

    hist_path = history_path(config, model=model)

    if network_path.exists() and not overwrite:
        print(f"Skip existing network: {network_path}")
        return None, None

    print(f"\nTraining indirect model {model}: D={PARAM_DIMS[model]}, N=384, S={config.summary_dim}\n")

    workflow = build_workflow(model=model, config=config)

    history = workflow.fit_online(
        epochs=config.epochs,
        batch_size=config.batch_size,
        num_batches_per_epoch=config.num_batches,
        validation_data=1000,
    )

    if save:
        network_path.parent.mkdir(parents=True, exist_ok=True)
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        workflow.approximator.save(network_path)

        history_data = {
            "method": "indirect",
            "model": model,
            "config": asdict(config),
            "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        }

        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2)

        print(f"Saved network: {network_path}")
        print(f"Saved history: {hist_path}")

    return workflow, history

def train_all(*, overwrite=False):
    for model, multiplier in itertools.product(MODELS, SUMMARY_MULTIPLIERS):
        summary_dim = PARAM_DIMS[model] * multiplier

        config = TrainingConfig(
            summary_dim=summary_dim,
            epochs=256,
            batch_size=64,
            num_batches=128,
            learning_rate=1e-4,
            seed=2025,
            summary_base_distribution=None,
        )

        train_one(model=model, config=config, overwrite=overwrite)

if __name__ == "__main__":
    train_all(overwrite=False)

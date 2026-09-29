"""Train indirect NPE models for the Gaussian case study."""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import itertools
import json
from dataclasses import asdict

import bayesflow as bf
import keras

from ..config import ASSUMED_MODELS
from .config import TrainingConfig, checkpoint_path, history_path
from .simulators import get_simulator


# General settings
MODELS = ASSUMED_MODELS
SUMMARY_DIMS = [20, 40,80]
NUM_OBS_VALUES = [ 10, 100]


def build_adapter():
    return (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .rename("mu", "inference_variables")
        .rename("x", "summary_variables")
    )

def build_workflow(model: str, config: TrainingConfig):

    keras.utils.set_random_seed(config.seed)

    simulator = get_simulator(model=model, config=config)
    adapter = build_adapter()
    summary_network = bf.networks.DeepSet(
        summary_dim=config.summary_dim, base_distribution=config.summary_base_distribution
    )
    inference_network = bf.networks.CouplingFlow()
    learning_rate = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=config.epochs * config.num_batches,
    )
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_network=summary_network,
        inference_network=inference_network,
        standardize="all",
        optimizer=optimizer,
    )
    return workflow

def train_one(model: str, config: TrainingConfig, *, save=True, overwrite=False):
    network_path = checkpoint_path(model, config)
    hist_path = history_path(model, config)

    # Skip training if the network already exists
    if network_path.exists() and not overwrite:
        print(f"Skip existing network: {network_path}")
        return None, None
    print(
        f"\nTraining indirect model {model}: "
        f"D={config.num_dims}, "
        f"N={config.num_obs}, "
        f"S={config.summary_dim}\n"
    )
    workflow = build_workflow(model, config)
    history = workflow.fit_online(
        epochs=config.epochs,
        batch_size=config.batch_size,
        num_batches_per_epoch=config.num_batches,
        validation_data=1000,
    )
    if save:
        network_path.parent.mkdir(parents=True, exist_ok=True,)
        hist_path.parent.mkdir(parents=True, exist_ok=True,)
        workflow.approximator.save(network_path)
        history_data = {
            "method": "indirect",
            "model": model,
            "config": asdict(config),
            "history": {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
        }
        with open(history_path, "w",encoding="utf-8",) as f:
            json.dump(history_data, f, indent=2,)

        print(f"Saved network: {network_path}")
        print(f"Saved history: {hist_path}")
    return workflow, history

def train_all(*, overwrite=False):
    for model, summary_dim, num_obs in itertools.product(
        MODELS, SUMMARY_DIMS, NUM_OBS_VALUES
    ):
        config = TrainingConfig(
            num_dims=20,
            num_obs=num_obs,
            summary_dim=summary_dim,
            epochs=256,
            batch_size=64,
            num_batches=128,
            summary_base_distribution="normal",
            learning_rate=1e-4,
            seed=2025,
        )
        train_one(model=model, config=config, overwrite=overwrite)


if __name__ == "__main__":
    train_all(overwrite=False)

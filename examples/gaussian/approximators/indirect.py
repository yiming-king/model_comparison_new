import os
import itertools

os.environ["KERAS_BACKEND"] = "tensorflow"

import bayesflow as bf
import keras
import json

from .config import TrainingConfig
from .simulators import get_simulator
from dataclasses import asdict
from ..config import NETWORK_DIR


# General settings
MODELS = [
    "m1",
    "m2",
    "m3",
    "m4",
]
SUMMARY_DIMS = [
    20,
    40,
    80,
]
NUM_OBS_VALUES = [
    10,
    100,
]


def get_name(model, config):
    return f"{model}_s_{config.summary_dim}d_{config.num_obs}n"


def get_paths(model, config):
    name = get_name(model, config)
    network_path = NETWORK_DIR / f"{name}.keras"
    history_path = NETWORK_DIR / "history" / f"{name}.json"
    return network_path, history_path


def build_adapter():
    return (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .rename("mu", "inference_variables")
        .rename("x", "summary_variables")
    )


def build_workflow(model, config):
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

def train_one(model, config, save=True, overwrite=False):
    network_path, history_path = get_paths(model, config)

    # Skip training if the network already exists
    if network_path.exists() and not overwrite:
        print(f"Skip existing network: {network_path}")
        return None, None
    print(
        f"\nTraining {model}: "
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
        network_path, history_path = get_paths(model, config)
        network_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        workflow.approximator.save(network_path)
        history_data = {
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
        print(f"Saved history: {history_path}")
    return workflow, history

def train_all(overwrite=False):
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
        train_one(model, config, overwrite=overwrite)


if __name__ == "__main__":
    train_all(overwrite=False)

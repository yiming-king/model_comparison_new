import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import itertools
import json

import bayesflow as bf
import keras

from dataclasses import asdict

from .config import TrainingConfig
from .simulators import get_simulator
from ..config import NETWORK_DIR

# General settings
MODELS = ["m1", "m2", "m3", "m4"]

SCORING_RULES = [
    "cross_entropy",
    "logistic",
    "exponential",
]

NUM_OBS_VALUES = [10, 100]


def get_name(scoring_rule: str, config):
    return f"direct_{scoring_rule}_{config.num_obs}n"


def get_paths(scoring_rule: str, config):
    name = get_name(scoring_rule, config)
    network_path = NETWORK_DIR / f"{name}.keras"
    history_path = NETWORK_DIR / "history" / f"{name}.json"
    return network_path, history_path


def get_simulators(config):
    return [
        get_simulator(model=model, config=config, seed=config.seed + i)
        for i, model in enumerate(MODELS)
    ]


def build_workflow(scoring_rule: str, config):
    workflow = bf.ModelComparisonWorkflow(
        simulator=get_simulators(config),
        summary_variables="x",
        summary_network=bf.networks.DeepSet(summary_dim=config.summary_dim),
        scoring_rules=scoring_rule,
        initial_learning_rate=config.learning_rate,
    )
    return workflow


def train_one(scoring_rule, config, save=True, overwrite=False):
    network_path, history_path = get_paths(scoring_rule, config)

    # Skip existing network unless overwrite is requested
    if save and network_path.exists() and not overwrite:
        print(f"Skip existing network: {network_path}")
        return None, None

    print(
        f"\nTraining direct model comparison: "
        f"scoring_rule={scoring_rule}, "
        f"N={config.num_obs}\n"
    )

    workflow = build_workflow(scoring_rule,config,)

    history = workflow.fit_online(
        epochs=config.epochs,
        batch_size=config.batch_size,
        num_batches_per_epoch=config.num_batches,
        validation_data=1000,
    )

    if save:
        network_path.parent.mkdir(parents=True, exist_ok=True,)
        history_path.parent.mkdir(parents=True, exist_ok=True,)

        workflow.approximator.save(network_path)

        history_data = {
            "scoring_rule": scoring_rule,
            "config": asdict(config),
            "history": {
                key: [float(v) for v in values]
                for key, values in history.history.items()
            },
        }

        with open(history_path, "w", encoding="utf-8",) as f:
            json.dump(history_data, f, indent=2,)

        print(f"Saved network: {network_path}")
        print(f"Saved history: {history_path}")

    return workflow, history

def train_all(overwrite=False):
    for scoring_rule, num_obs in itertools.product(SCORING_RULES, NUM_OBS_VALUES):
        config = TrainingConfig(
            num_dims=20,
            num_obs=num_obs,
            summary_dim=12,
            epochs=256,
            batch_size=64,
            num_batches=128,
            learning_rate=1e-4,
            seed=2025,
        )
        train_one(scoring_rule, config, save=True, overwrite=overwrite)


if __name__ == "__main__":
    train_all(overwrite=False)
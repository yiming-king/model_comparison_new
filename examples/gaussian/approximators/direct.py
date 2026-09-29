import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import itertools
import json
from dataclasses import asdict

import bayesflow as bf
import keras

from ..config import ASSUMED_MODELS
from .config import (
    DIRECT_SCORING_RULES,
    TrainingConfig,
    checkpoint_path,
    history_path,
)
from .simulators import get_simulator
# General settings
MODELS = ASSUMED_MODELS

SCORING_RULES = DIRECT_SCORING_RULES
NUM_OBS_VALUES = [10, 100]


def get_simulators(config: TrainingConfig):
    return [
        get_simulator(model=model, config=config, seed=config.seed + i)
        for i, model in enumerate(MODELS)
    ]


def build_workflow(scoring_rule: str, config: TrainingConfig):
    keras.utils.set_random_seed(config.seed)
    workflow = bf.ModelComparisonWorkflow(
        simulator=get_simulators(config),
        summary_variables="x",
        summary_network=bf.networks.DeepSet(summary_dim=config.summary_dim),
        scoring_rules=scoring_rule,
        initial_learning_rate=config.learning_rate,
    )
    return workflow



def train_one(scoring_rule: str, config: TrainingConfig, *, save=True, overwrite=False):
    network_path = checkpoint_path(config, scoring_rule=scoring_rule)
    hist_path = history_path(config, scoring_rule=scoring_rule)

    # Skip existing network unless overwrite is requested
    if network_path.exists() and not overwrite:
        print(f"Skip existing network: {network_path}")
        return None, None

    print(
        f"\nTraining direct model comparison\n"
        f"loss={scoring_rule}\n"
        f"D={config.num_dims}\n"
        f"N={config.num_obs}\n"
        f"S={config.summary_dim}\n"
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
        hist_path.parent.mkdir(parents=True, exist_ok=True,)

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

def train_all(*, overwrite=False):
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
            summary_base_distribution=None,
        )
        train_one(scoring_rule=scoring_rule, config=config, overwrite=overwrite)


if __name__ == "__main__":
    train_all(overwrite=False)
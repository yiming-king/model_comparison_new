"""Train direct model-comparison networks for the diffusion case study."""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import json
from dataclasses import asdict

import bayesflow as bf
import keras

from ..config import ASSUMED_MODELS
from ..simulators import SIMULATORS_NO_PARAMS
from .config import DIRECT_SCORING_RULES, TrainingConfig, checkpoint_path, history_path

MODELS = ASSUMED_MODELS
SCORING_RULES = DIRECT_SCORING_RULES

SUMMARY_DIM = 12

def build_adapter():
    return (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .as_set(["rt", "conditions"])
        .concatenate(["rt", "conditions"], into="summary_variables")
        .rename("model_indices", "inference_variables")
    )

def get_simulators():
    return [SIMULATORS_NO_PARAMS[model] for model in MODELS]

def build_workflow(scoring_rule: str, config: TrainingConfig):
    keras.utils.set_random_seed(config.seed)

    return bf.ModelComparisonWorkflow(
        simulator=get_simulators(),
        adapter=build_adapter(),
        summary_network=bf.networks.DeepSet(summary_dim=config.summary_dim),
        scoring_rules=scoring_rule,
        model_names=[model.upper() for model in MODELS],
        initial_learning_rate=config.learning_rate,
    )

def train_one(scoring_rule: str, config: TrainingConfig, *, save=True, overwrite=False):
    network_path = checkpoint_path(config, scoring_rule=scoring_rule)

    hist_path = history_path(config, scoring_rule=scoring_rule)

    if network_path.exists() and not overwrite:
        print(f"Skip existing network: {network_path}")
        return None, None

    print(f"\nTraining direct model comparison\nloss={scoring_rule}\nN=384\nS={config.summary_dim}\n")

    workflow = build_workflow(scoring_rule=scoring_rule, config=config)

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
            "method": "direct",
            "scoring_rule": scoring_rule,
            "config": asdict(config),
            "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        }

        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2)

        print(f"Saved network: {network_path}")
        print(f"Saved history: {hist_path}")

    return workflow, history

def train_all(*, overwrite=False):
    for scoring_rule in SCORING_RULES:
        config = TrainingConfig(
            summary_dim=SUMMARY_DIM, epochs=256, batch_size=64, num_batches=128, learning_rate=1e-4, seed=2025
        )

        train_one(scoring_rule=scoring_rule, config=config, overwrite=overwrite)

if __name__ == "__main__":
    train_all(overwrite=False)

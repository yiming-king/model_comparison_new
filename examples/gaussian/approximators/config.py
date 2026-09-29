"""Shared training settings for the Gaussian observation/summary comparison."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path

from ..config import ASSUMED_MODELS, NETWORK_DIR


DIRECT_SCORING_RULES = ("cross_entropy", "exponential", "logistic")


def validate_scoring_rule(scoring_rule: str) -> None:
    if scoring_rule not in DIRECT_SCORING_RULES:
        raise ValueError(f"Unknown direct loss: {scoring_rule}")


# Values are (raw data/parameter dimension, observations, summary dimension).
# Preset tags use the summary dimension, followed by the observation count.
# Includes the historical notebook dimensions and the two missing N=100 cases.
NOTEBOOK_PRESETS = {
    "20d_10n": (20, 10, 20),
    "40d_10n": (20, 10, 40),
    "80d_10n": (20, 10, 80),
    "20d_100n": (20, 100, 20),
    "40d_100n": (20, 100, 40),
    "80d_100n": (20, 100, 80),
}


def validate_model(model: str) -> None:
    if model not in ASSUMED_MODELS:
        raise ValueError(
            f"Unknown assumed model: {model}; choose from {ASSUMED_MODELS}"
        )


def positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_seed(name: str, value: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= value < 2**32
    ):
        raise ValueError(f"{name} must be an integer between 0 and 2**32 - 1")


@dataclass(frozen=True)
class TrainingConfig:
    num_dims: int = 20
    num_obs: int = 10
    summary_dim: int = 20
    epochs: int = 256
    batch_size: int = 64
    num_batches: int = 128
    learning_rate: float = 1e-4
    seed: int = 2025
    summary_base_distribution: str | None = None
    run_suffix: str | None = None

    def __post_init__(self) -> None:
        for name in ("num_dims", "num_obs", "summary_dim", "batch_size", "num_batches"):
            positive_int(name, getattr(self, name))
        if self.epochs is not None:
            positive_int("epochs", self.epochs)
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        validate_seed("seed", self.seed)
        if self.summary_base_distribution not in (None, "normal"):
            raise ValueError("summary_base_distribution must be 'normal' or None")
        if self.run_suffix is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_suffix
        ):
            raise ValueError(
                "run_suffix must start with a letter or digit and contain only letters, digits, _, . or -"
            )

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> TrainingConfig:
        try:
            num_dims, num_obs, summary_dim = NOTEBOOK_PRESETS[preset]
        except KeyError as error:
            raise ValueError(f"Unknown training preset: {preset}") from error
        return replace(
            cls(num_dims=num_dims, num_obs=num_obs, summary_dim=summary_dim),
            **overrides,
        )

    def epochs_for(self, model: str) -> int:
        validate_model(model)
        if self.epochs is not None:
            return self.epochs
        return 100

    @property
    def network_tag(self) -> str:
        dimensions = (self.num_dims, self.num_obs, self.summary_dim)
        tag = next(
            (name for name, values in NOTEBOOK_PRESETS.items() if values == dimensions),
            f"{self.num_dims}d_{self.num_obs}n_s{self.summary_dim}",
        )
        # Keep the unified network settings separate from historical notebooks.
        # In particular, never silently reuse their same-dimension model files.
        tag += "_bf_default"
        if self.summary_base_distribution is not None:
            tag += "_mmd"
        return f"{tag}_{self.run_suffix}" if self.run_suffix else tag


def model_path(
    model: str,
    config: TrainingConfig = TrainingConfig(),
    output_dir: str | Path = NETWORK_DIR,
    *,
    scoring_rule: str | None = None,
) -> Path:
    """Single checkpoint naming rule for Gaussian NPE and direct training."""
    if model == "direct":
        validate_scoring_rule(scoring_rule)
        model = f"direct_{scoring_rule}"
    else:
        validate_model(model)
        if scoring_rule is not None:
            raise ValueError("scoring_rule is only applicable to direct models")
    return Path(output_dir) / f"{model}_s_{config.network_tag}.keras"

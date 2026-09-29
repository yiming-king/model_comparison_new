"""Training configuration and checkpoint paths."""

from dataclasses import dataclass
from numbers import Integral

from ..config import ASSUMED_MODELS, NETWORK_DIR


DIRECT_SCORING_RULES = (
    "cross_entropy",
    "logistic",
    "exponential",
)


def validate_model(model: str) -> None:
    if model not in ASSUMED_MODELS:
        raise ValueError(
            f"Unknown model: {model}. "
            f"Choose from {ASSUMED_MODELS}."
        )


def validate_scoring_rule(scoring_rule: str) -> None:
    if scoring_rule not in DIRECT_SCORING_RULES:
        raise ValueError(
            f"Unknown scoring rule: {scoring_rule}. "
            f"Choose from {DIRECT_SCORING_RULES}."
        )


def validate_seed(seed: int) -> None:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or not 0 <= seed < 2**32
    ):
        raise ValueError(
            "seed must be an integer between 0 and 2**32 - 1"
        )


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

    def __post_init__(self) -> None:
        if self.num_dims <= 0:
            raise ValueError("num_dims must be positive")

        if self.num_obs <= 0:
            raise ValueError("num_obs must be positive")

        if self.summary_dim <= 0:
            raise ValueError("summary_dim must be positive")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.num_batches <= 0:
            raise ValueError("num_batches must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        validate_seed(self.seed)

        if self.summary_base_distribution not in (
            None,
            "normal",
        ):
            raise ValueError(
                "summary_base_distribution must be "
                "None or 'normal'"
            )

    @property
    def network_tag(self) -> str:
        """Identifier shared by checkpoint and history files."""

        tag = (
            f"n{self.num_obs}"
            f"_s{self.summary_dim}"
        )

        if self.summary_base_distribution == "normal":
            tag += "_mmd"

        return tag


def run_name(
    config: TrainingConfig,
    *,
    model: str | None = None,
    scoring_rule: str | None = None,
) -> str:
    """Return one unique name for an indirect or direct run."""

    if (model is None) == (scoring_rule is None):
        raise ValueError(
            "Provide either model or scoring_rule, but not both."
        )

    if model is not None:
        validate_model(model)
        prefix = model

    else:
        validate_scoring_rule(scoring_rule)
        prefix = f"direct_{scoring_rule}"

    return f"{prefix}_{config.network_tag}"


def checkpoint_path(
    config: TrainingConfig,
    *,
    model: str | None = None,
    scoring_rule: str | None = None,
):
    """Return the .keras checkpoint path."""

    name = run_name(
        config,
        model=model,
        scoring_rule=scoring_rule,
    )

    return NETWORK_DIR / f"{name}.keras"


def history_path(
    config: TrainingConfig,
    *,
    model: str | None = None,
    scoring_rule: str | None = None,
):
    """Return the training-history JSON path."""

    name = run_name(
        config,
        model=model,
        scoring_rule=scoring_rule,
    )

    return NETWORK_DIR / "history" / f"{name}.json"
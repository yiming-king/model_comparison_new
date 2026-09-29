"""Shared configuration for the Gaussian case study."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
NETWORK_DIR = BASE_DIR / "networks"
RESULT_DIR = BASE_DIR / "results"
CALIBRATION_ROOT = BASE_DIR / "calibration_outputs"


def calibration_output_dir(
    configuration: str, root: str | Path = CALIBRATION_ROOT
) -> Path:
    """Keep the primary 20d_10n calibration directly in calibration_outputs."""
    root = Path(root)
    return root if configuration == "20d_10n" else root / configuration

MODEL_SPECS = {
    "m1": {"mu_prior_mean": 0.0, "mu_prior_std": 1.0, "likelihood_std": 1.0},
    "m2": {"mu_prior_mean": 3.0, "mu_prior_std": 1.0, "likelihood_std": 1.0},
    "m3": {"mu_prior_mean": 0.0, "mu_prior_std": 1.0, "likelihood_std": 3.0},
    "m4": {"mu_prior_mean": 0.1, "mu_prior_std": 1.0, "likelihood_std": 1.0},
    "m5": {"mu_prior_mean": 1.5, "mu_prior_std": 1.0, "likelihood_std": 1.0},
    "m6": {"mu_prior_mean": 5.0, "mu_prior_std": 1.0, "likelihood_std": 1.0},
    "m7": {"mu_prior_mean": 0.0, "mu_prior_std": 1.0, "likelihood_std": 5.0},
    "m8": {"mu_prior_mean": 0.0, "mu_prior_std": 3.0, "likelihood_std": 1.0},
    "m9": {"mu_prior_mean": 0.0, "mu_prior_std": 1.0, "likelihood_std": 0.1},
    "m10": {"mu_prior_mean": 0.0, "mu_prior_std": 1.0, "likelihood_std": 0.01},
    "m11": {"mu_prior_mean": 0.0, "mu_prior_std": 0.1, "likelihood_std": 1.0},
    "m12": {"mu_prior_mean": 0.0, "mu_prior_std": 0.01, "likelihood_std": 1.0},
}

ASSUMED_MODELS = tuple(MODEL_SPECS)[:4]
SOURCE_MODELS = tuple(MODEL_SPECS)


def configuration_tag(num_dims: int, num_obs: int) -> str:
    """Return the tag used by trained-network and result filenames."""
    if num_dims <= 0 or num_obs <= 0:
        raise ValueError("num_dims and num_obs must be positive")
    return f"{int(num_dims)}d_{int(num_obs)}n"


def network_path(model: str, num_dims: int, num_obs: int) -> Path:
    if model not in ASSUMED_MODELS:
        raise ValueError(f"Unknown assumed model: {model}")
    return NETWORK_DIR / f"{model}_s_{configuration_tag(num_dims, num_obs)}.keras"


@dataclass(frozen=True)
class NetworkSet:
    """One matched m1--m4 network set and its actual summary-space size."""

    network_tag: str
    data_dim: int
    num_obs: int
    summary_dim: int
    paths: dict[str, Path]

    @property
    def summary_multiplier(self) -> float:
        return self.summary_dim / self.data_dim

    @property
    def summary_label(self) -> str:
        multiplier = self.summary_multiplier
        if multiplier.is_integer():
            return f"S={int(multiplier)}D"
        return f"S={multiplier:g}D"

    @property
    def summary_slug(self) -> str:
        """Return the filesystem label used by diffusion-style metric folders."""
        return self.summary_label.replace("=", "")


_NETWORK_NAME = re.compile(r"^(m[1-4])_s_(.+)\.keras$")


def _find_continuous_approximator(config: object) -> dict:
    if isinstance(config, dict):
        if config.get("class_name") == "ContinuousApproximator":
            return config
        for value in config.values():
            found = _find_continuous_approximator(value)
            if found is not None:
                return found
    elif isinstance(config, list):
        for value in config:
            found = _find_continuous_approximator(value)
            if found is not None:
                return found
    return None


def read_network_set_metadata(path: str | Path) -> tuple[int, int, int]:
    """Read raw input D/n and true summary dimension without loading Keras."""
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
    approximator = _find_continuous_approximator(config)
    if approximator is None:
        raise ValueError(f"No ContinuousApproximator configuration in {path}")
    input_shape = approximator["build_config"]["input_shape"]["summary_variables"]
    summary_dim = approximator["config"]["summary_network"]["config"]["summary_dim"]
    if len(input_shape) != 3 or input_shape[1] is None or input_shape[2] is None:
        raise ValueError(f"Unexpected summary input shape in {path}: {input_shape}")
    return int(input_shape[2]), int(input_shape[1]), int(summary_dim)


def discover_network_sets(
    network_dir: str | Path = NETWORK_DIR,
    *,
    data_dim: int | None = None,
    num_obs: int | None = None,
    summary_dims: tuple[int, ...] | None = None,
    network_tags: tuple[str, ...] | None = None,
) -> tuple[NetworkSet, ...]:
    """Discover complete m1--m4 network sets and group by real summary size."""
    network_dir = Path(network_dir)
    grouped: dict[tuple[str, int, int, int], dict[str, Path]] = {}
    for path in network_dir.glob("m[1-4]_s_*.keras"):
        match = _NETWORK_NAME.match(path.name)
        if match is None:
            continue
        model, tag = match.groups()
        raw_dim, observations, summary_dim = read_network_set_metadata(path)
        if data_dim is not None and raw_dim != data_dim:
            continue
        if num_obs is not None and observations != num_obs:
            continue
        if summary_dims is not None and summary_dim not in summary_dims:
            continue
        if network_tags is not None and tag not in network_tags:
            continue
        grouped.setdefault((tag, raw_dim, observations, summary_dim), {})[model] = path

    sets = []
    for (tag, raw_dim, observations, summary_dim), paths in grouped.items():
        missing = set(ASSUMED_MODELS).difference(paths)
        if missing:
            continue
        sets.append(NetworkSet(tag, raw_dim, observations, summary_dim, dict(paths)))
    return tuple(
        sorted(
            sets,
            key=lambda item: (
                item.data_dim,
                item.num_obs,
                item.summary_dim,
                item.network_tag,
            ),
        )
    )

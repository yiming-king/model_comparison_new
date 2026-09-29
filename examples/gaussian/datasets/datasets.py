from pathlib import Path

import numpy as np


class GaussianDatasetGenerator:
    def __init__(
        self,
        mu_prior_mean: float,
        mu_prior_std: float,
        num_dims: int,
        num_obs: int,
        likelihood_std: float,
        rng=None,
    ):
        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_std = mu_prior_std
        self.num_dims = num_dims
        self.num_obs = num_obs
        self.likelihood_std = likelihood_std
        self.rng = rng if rng is not None else np.random.default_rng()

    def simulate(self, num_datasets: int):
        """Simulate datasets from the Gaussian generative model."""

        mu = self.rng.normal(
            loc=self.mu_prior_mean,
            scale=self.mu_prior_std,
            size=(num_datasets, self.num_dims),
        )

        x = self.rng.normal(
            loc=mu[:, None, :],
            scale=self.likelihood_std,
            size=(num_datasets, self.num_obs, self.num_dims),
        )

        return {
            "mu": mu,
            "x": x,
            "id": np.arange(num_datasets),
        }


def save_datasets(datasets: dict, path):
    """Save simulated datasets as a compressed NumPy file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(path, **datasets)


def load_datasets(path):
    """Load datasets saved with save_datasets."""

    with np.load(path) as data:
        return {key: data[key] for key in data.files}
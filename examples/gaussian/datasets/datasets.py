"""Generate, save, and load Gaussian datasets."""

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
        self.rng = (
            rng
            if rng is not None
            else np.random.default_rng()
        )

    def simulate(
        self,
        num_datasets: int,
        source_model: str,
    ) -> dict[str, np.ndarray]:
        """Simulate a batch of Gaussian datasets."""

        mu = self.rng.normal(loc=self.mu_prior_mean, scale=self.mu_prior_std,size=(num_datasets,self.num_dims,),)
        x = self.rng.normal(loc=mu[:, None, :], scale=self.likelihood_std, size=(num_datasets, self.num_obs, self.num_dims,),)

        return {
            "mu": mu,
            "x": x,
            "id": np.arange(num_datasets),
            "source_model": np.full(
                num_datasets,
                source_model,
            ),
        }


def save_datasets(
    datasets: dict[str, np.ndarray],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Dataset already exists: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        **datasets,
    )


def load_datasets(
    path: str | Path,
) -> dict[str, np.ndarray]:
    path = Path(path)

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        return {
            key: data[key]
            for key in data.files
        }
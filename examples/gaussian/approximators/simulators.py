import bayesflow as bf
import numpy as np

from ..config import MODEL_SPECS
from .config import TrainingConfig, validate_model, validate_seed


def get_simulator(
    model: str,
    config: TrainingConfig = TrainingConfig(),
    seed: int | None = None,
):
    validate_model(model)
    seed = config.seed if seed is None else seed
    validate_seed(seed)

    rng = np.random.default_rng(seed)
    spec = MODEL_SPECS[model]

    def prior():
        return {
            "mu": rng.normal(
                spec["mu_prior_mean"], spec["mu_prior_std"], config.num_dims
            )
        }

    def likelihood(mu):
        return {
            "x": rng.normal(
                loc=mu,
                scale=spec["likelihood_std"],
                size=(config.num_obs, config.num_dims),
            )
        }

    return bf.make_simulator([prior, likelihood])

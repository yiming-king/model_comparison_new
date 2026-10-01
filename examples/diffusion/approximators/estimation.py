"""Marginal-likelihood estimation for diffusion NPE models."""

import keras
import numpy as np
from scipy.special import logsumexp

from ..config import N_ALPHA
from ..distributions import Likelihood, Prior

def parameter_array(samples):
    """Convert posterior samples to one parameter array."""
    if "inference_variables" in samples:
        return np.asarray(samples["inference_variables"], dtype=np.float32)
    return np.concatenate(
        [
            np.asarray(samples["alpha"], dtype=np.float32),
            np.asarray(samples["nu"], dtype=np.float32),
            np.asarray(samples["tau"], dtype=np.float32),
        ],
        axis=-1,
    )

def parameter_dict(theta, model):
    """Split one parameter array into the RDM parameter blocks."""
    theta = np.asarray(theta, dtype=np.float32)
    n_alpha = N_ALPHA[model]
    return {
        "alpha": theta[..., :n_alpha],
        "nu": theta[..., n_alpha : n_alpha + 2],
        "tau": theta[..., n_alpha + 2 : n_alpha + 3],
    }

class MarginalLikelihoodEstimator:
    def __init__(self, approximator, theta, data, model, batch_size=None):
        self.approximator = approximator
        self.theta = np.asarray(theta, dtype=np.float32)
        self.data = np.asarray(data, dtype=np.float32)
        self.model = model
        self.batch_size = batch_size

    def log_prior(self):
        values = Prior(model=self.model).log_prob(keras.ops.convert_to_tensor(self.theta))
        return np.asarray(keras.ops.convert_to_numpy(values), dtype=np.float64).reshape(-1)

    def log_likelihood(self):
        data = np.repeat(self.data[None, :, :], repeats=len(self.theta), axis=0)
        values = Likelihood(model=self.model).log_prob(
            samples=keras.ops.convert_to_tensor(data), conditions=keras.ops.convert_to_tensor(self.theta)
        )
        return np.asarray(keras.ops.convert_to_numpy(values), dtype=np.float64).reshape(-1)

    def log_q(self):
        parameters = parameter_dict(self.theta, self.model)
        parameters["rt"] = np.repeat(self.data[None, :, 0], repeats=len(self.theta), axis=0)
        parameters["conditions"] = np.repeat(self.data[None, :, 1], repeats=len(self.theta), axis=0)
        values = self.approximator.log_prob(parameters, batch_size=self.batch_size)
        return np.asarray(values, dtype=np.float64).reshape(-1)

    def log_marginal_npe(self):
        log_weights = self.log_prior() + self.log_likelihood() - self.log_q()
        normalized_log_weights = log_weights - logsumexp(log_weights)
        self.importance_ess = float(np.exp(-logsumexp(2.0 * normalized_log_weights)))
        return float(logsumexp(log_weights) - np.log(len(log_weights)))

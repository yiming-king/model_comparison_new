from __future__ import annotations

import keras.ops as ops
from bayesflow.distributions.distribution import Distribution
from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable, serialize

from ..config import N_ALPHA
from .normal import normal_lpdf
from .utils import expit
from .wald import rdm_lpdf

def _alpha_by_trial(parameters: Tensor, condition: Tensor, model: str) -> Tensor:
    if model == "m0":
        alpha = ops.exp(parameters[..., 0])
        return ops.stack([alpha, alpha], axis=-1)

    if model == "m1":
        alpha = ops.where(condition > 0.5, ops.exp(parameters[..., 1]), ops.exp(parameters[..., 0]))
        return ops.stack([alpha, alpha], axis=-1)

    if model == "m2":
        return ops.exp(parameters[..., 0:2])

    alpha0 = ops.exp(parameters[..., 0:2])
    alpha1 = ops.exp(parameters[..., 2:4])
    return ops.where(ops.expand_dims(condition > 0.5, axis=-1), alpha1, alpha0)

@serializable("bayesflow.distributions")
class Prior(Distribution):
    def __init__(self, model: str = "m0", **kwargs):
        super().__init__(**kwargs)
        self.built = True
        self.model = model

    def get_config(self):
        return serialize({"model": self.model})

    def log_prob(self, samples: Tensor, conditions: Tensor | None = None) -> Tensor:
        samples = ops.cast(samples, "float32")
        n_alpha = N_ALPHA[self.model]
        alpha = ops.sum(normal_lpdf(samples[:, :n_alpha], mu=0.0, sigma=0.5), axis=-1)
        nu = ops.sum(normal_lpdf(samples[:, n_alpha : n_alpha + 2], mu=0.0, sigma=0.5), axis=-1)
        tau = normal_lpdf(samples[:, n_alpha + 2], mu=0.0, sigma=1.0)
        return alpha + nu + tau

@serializable("bayesflow.distributions")
class Likelihood(Distribution):
    def __init__(self, model: str = "m0", **kwargs):
        super().__init__(**kwargs)
        self.built = True
        self.model = model

    def get_config(self):
        return serialize({"model": self.model})

    def log_prob(self, samples: Tensor, conditions: Tensor) -> Tensor:
        conditions = ops.cast(conditions, "float64")
        samples = ops.cast(samples, conditions.dtype)
        rt = samples[..., 0]
        condition = samples[..., 1]

        parameters = ops.expand_dims(conditions, axis=1)
        parameters = ops.repeat(parameters, ops.shape(rt)[1], axis=1)

        n_alpha = N_ALPHA[self.model]
        alpha = _alpha_by_trial(parameters, condition, self.model)
        nu = ops.exp(parameters[..., n_alpha : n_alpha + 2])
        tau = expit(parameters[..., n_alpha + 2])
        trial_log_prob = rdm_lpdf(rt=rt, alpha=alpha, nu=nu, tau=tau)
        return ops.sum(trial_log_prob, axis=-1) + ops.log1p(-tau[:, 0])

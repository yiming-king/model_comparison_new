from __future__ import annotations

import bayesflow as bf
import numpy as np
from bayesflow.types import Shape
from bayesflow.utils.decorators import allow_batch_size
from scipy.special import expit

from ..config import MODELS, N_ALPHA
from ..dataset import wagenmakers


class RDM(bf.simulators.Simulator):
    """Racing diffusion simulator for the four threshold structures used in Stan."""

    def __init__(self, model: str, keep_params: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.keep_params = keep_params

    @allow_batch_size
    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        parameters = self.prior(batch_shape)
        observables = self.likelihood(
            batch_shape, **self._constrain_parameters(**parameters)
        )
        return parameters | observables if self.keep_params else observables

    @allow_batch_size
    def prior(self, batch_shape: Shape) -> dict[str, np.ndarray]:
        return {
            "alpha": np.random.normal(
                0.0, 0.5, size=batch_shape + (N_ALPHA[self.model],)
            ),
            "nu": np.random.normal(0.0, 0.5, size=batch_shape + (2,)),
            "tau": np.random.normal(0.0, 1.0, size=batch_shape + (1,)),
        }

    @staticmethod
    def _constrain_parameters(
        alpha: np.ndarray, nu: np.ndarray, tau: np.ndarray
    ) -> dict[str, np.ndarray]:
        return {"alpha": np.exp(alpha), "nu": np.exp(nu), "tau": expit(tau)}

    @allow_batch_size
    def likelihood(
        self, batch_shape: Shape, alpha: np.ndarray, nu: np.ndarray, tau: np.ndarray
    ) -> dict[str, np.ndarray]:
        conditions = wagenmakers.conditions.reshape((1,) * len(batch_shape) + (-1,))
        conditions = np.broadcast_to(conditions, batch_shape + (conditions.shape[-1],))
        rt = self.rdm_rng(
            alpha=self._alpha_by_trial(alpha, conditions),
            nu=np.expand_dims(nu, axis=1),
            tau=tau,
        )
        return {"rt": rt, "conditions": conditions}

    def _alpha_by_trial(self, alpha: np.ndarray, conditions: np.ndarray) -> np.ndarray:
        if self.model == "m0":
            pair = np.repeat(alpha[..., None, :], 2, axis=-1)
            return np.repeat(pair, conditions.shape[-1], axis=-2)

        if self.model == "m1":
            selected = np.where(
                conditions.astype(bool), alpha[..., 1, None], alpha[..., 0, None]
            )
            return np.repeat(selected[..., None], 2, axis=-1)

        if self.model == "m2":
            pair = alpha[..., None, :]
            return np.repeat(pair, conditions.shape[-1], axis=-2)

        condition0 = np.repeat(alpha[..., None, 0:2], conditions.shape[-1], axis=-2)
        condition1 = np.repeat(alpha[..., None, 2:4], conditions.shape[-1], axis=-2)
        return np.where(conditions.astype(bool)[..., None], condition1, condition0)

    def rdm_rng(self, alpha: np.ndarray, nu: np.ndarray, tau: np.ndarray) -> np.ndarray:
        decision_times = self.wald_rng(alpha=alpha, nu=nu)
        t = np.min(decision_times, axis=-1)
        choice = np.argmin(decision_times, axis=-1)

        min_t = np.min(t, axis=-1, keepdims=True)
        t0 = tau * min_t / (1.0 - tau)
        rt = t + t0
        rt[choice == 1] *= -1.0
        return rt

    @staticmethod
    def wald_rng(alpha: np.ndarray, nu: np.ndarray) -> np.ndarray:
        mu = alpha / nu
        mu_sq = np.square(mu)
        lam = np.square(alpha)
        zeta_sq = np.square(np.random.standard_normal(size=mu.shape))

        x = (
            mu
            + (mu_sq * zeta_sq) / (2.0 * lam)
            - mu
            / (2.0 * lam)
            * np.sqrt(4.0 * mu * lam * zeta_sq + mu_sq * np.square(zeta_sq))
        )
        z = np.random.uniform(size=mu.shape)
        return np.where(z <= mu / (mu + x), x, mu_sq / x)


SIMULATORS = {model: RDM(model=model) for model in MODELS}
SIMULATORS_NO_PARAMS = {model: RDM(model=model, keep_params=False) for model in MODELS}
rdm_model_comparison = bf.simulators.ModelComparisonSimulator(
    list(SIMULATORS_NO_PARAMS.values())
)

rdm_m0 = SIMULATORS["m0"]
rdm_m1 = SIMULATORS["m1"]
rdm_m2 = SIMULATORS["m2"]
rdm_m3 = SIMULATORS["m3"]


if __name__ == "__main__":
    for model, simulator in SIMULATORS.items():
        print(f"\n{model}\n")
        data = simulator.sample((4,))
        for key, value in data.items():
            print(key, value.shape)

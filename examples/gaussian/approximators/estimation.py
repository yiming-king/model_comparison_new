import numpy as np
from scipy.special import logsumexp


def logmeanexp(log_terms: np.ndarray) -> float:
    log_sum_exp = logsumexp(log_terms) - np.log(len(log_terms))
    return log_sum_exp


"use the prior of training datasets"


class MarginalLikelihoodEstimator:
    def __init__(
        self,
        approximator,
        mu: np.ndarray,
        obs_data: np.ndarray,
        mu_prior_mean: float,
        mu_prior_std: float,
        num_dims: int,
        likelihood_std: float,
        df: float | None = None,
        rng=None,
    ):
        """mu is the estimated posterior samples"""
        self.approximator = approximator
        self.mu = mu
        self.obs_data = obs_data
        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_std = mu_prior_std
        self.num_dims = num_dims
        self.likelihood_std = likelihood_std
        self.df = df
        self.rng = rng if rng is not None else np.random.default_rng()

    def log_prior_mu(self):
        mu = np.asarray(self.mu)
        mu0 = float(self.mu_prior_mean)
        tau2 = float(self.mu_prior_std**2)
        D = self.num_dims

        log_prior = -0.5 * (
            D * np.log(2 * np.pi * tau2) + np.sum((mu - mu0) ** 2, axis=1) / tau2
        )
        return log_prior

    def log_likelihood_x_given_mu(self):
        "mu: shape (num_samples,num_dims), x_obs: shape (num_obs,num_dims)"
        mu = np.asarray(self.mu)
        obs_data = np.asarray(self.obs_data)
        sig2 = float(self.likelihood_std**2)  # likelihood variance
        diff = obs_data[None, :, :] - mu[:, None, :]
        const = -0.5 * np.log(2 * np.pi * sig2)
        logpdf = const - 0.5 * (diff**2) / sig2
        return np.sum(logpdf, axis=(1, 2))

    def log_q_phi(self):
        mu = np.asarray(self.mu)
        obs_data = np.asarray(self.obs_data)
        S = mu.shape[0]
        data = {"x": np.repeat(obs_data[None, :, :], S, axis=0), "mu": mu}
        log_probs = self.approximator.log_prob(data)
        log_probs = np.asarray(log_probs)
        return log_probs.reshape(-1)

    def log_marginal_npe(self, method: str = "log_mean_exp") -> float:
        """Estimate log marginal likelihood using log-mean-exp posterior weights."""
        if method != "log_mean_exp":
            raise ValueError("method must be 'log_mean_exp'")
        log_terms = (
            self.log_prior_mu() + self.log_likelihood_x_given_mu() - self.log_q_phi()
        )
        normalized_log_weights = log_terms - logsumexp(log_terms)
        self.importance_ess = float(np.exp(-logsumexp(2.0 * normalized_log_weights)))
        self.num_importance_samples = int(len(log_terms))

        return float(logmeanexp(log_terms))

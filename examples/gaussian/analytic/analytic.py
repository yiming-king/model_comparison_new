import numpy as np


class GaussianAnalytical:
    def __init__(
        self,
        obs_data: np.ndarray,
        mu_prior_mean: float | np.ndarray,
        mu_prior_std: float,
        num_dims: int,
        num_obs: int,
        num_samples: int,
        likelihood_std: float,
        rng=None,
    ):
        """Initialize the analytical posterior."""
        self.obs_data = obs_data
        self.mu_prior_std = mu_prior_std
        self.num_dims = num_dims
        self.num_obs = num_obs
        self.num_samples = num_samples
        self.likelihood_std = likelihood_std
        self.rng = rng if rng is not None else np.random.default_rng()
        mu_prior_mean = np.asarray(mu_prior_mean, dtype=float)
        if mu_prior_mean.ndim == 0:
            mu_prior_mean = np.full(num_dims, float(mu_prior_mean))
        elif mu_prior_mean.shape != (num_dims,):
            raise ValueError(
                f"mu_prior_mean must have shape ({num_dims},), got {mu_prior_mean.shape}"
            )

        if obs_data.shape != (num_obs, num_dims):
            raise ValueError(
                f"obs_data must have shape ({num_obs}, {num_dims}), got {obs_data.shape}"
            )

        self.mu_prior_mean = mu_prior_mean

    def analytical_posterior(self) -> tuple[np.ndarray, np.ndarray]:

        x_bar = self.obs_data.mean(axis=0)  # (D,)
        mu_post_var = 1 / (
            self.num_obs / (self.likelihood_std**2) + 1 / (self.mu_prior_std**2)
        )
        mu_post_mean = mu_post_var * (
            self.num_obs * x_bar / (self.likelihood_std**2)
            + 1 * self.mu_prior_mean / (self.mu_prior_std**2)
        )  # (D,)
        analytical_posterior_samples = self.rng.normal(
            loc=mu_post_mean,
            scale=np.sqrt(mu_post_var),
            size=(self.num_samples, self.num_dims),
        )

        return analytical_posterior_samples

    def log_marginal_analytical(self) -> float:
        """Compute the analytical log marginal likelihood."""
        sig2 = self.likelihood_std**2
        tau2 = self.mu_prior_std**2
        mu0 = self.mu_prior_mean  # (D,)

        logdet = (self.num_obs - 1) * np.log(sig2) + np.log(sig2 + self.num_obs * tau2)

        xc = self.obs_data - mu0
        sum_sq = np.sum(xc**2, axis=0)
        sum_ = np.sum(xc, axis=0)
        quad = (1.0 / sig2) * sum_sq - (
            tau2 / (sig2 * (sig2 + self.num_obs * tau2))
        ) * (sum_**2)  # (D,)

        logp_per_dim = -0.5 * (self.num_obs * np.log(2 * np.pi) + logdet + quad)
        return float(np.sum(logp_per_dim))

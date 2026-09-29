import numpy as np
from examples.gaussian.analytic.analytic import GaussianAnalytical as GA
from examples.gaussian.approximators.estimation import (
    MarginalLikelihoodEstimator as MLE,
)


class Calculation:
    def __init__(
        self,
        approximator,
        mu_prior_mean: float,
        mu_prior_std: float,
        num_dims: int,
        num_obs: int,
        likelihood_std: float,
        num_samples: int,
        assumed_model: str,
        df: float | None = None,
        use_student_t: bool = False,
        rng=None,
        logml_method: str = "log_mean_exp",
    ):
        if logml_method != "log_mean_exp":
            raise ValueError("logml_method must be 'log_mean_exp'")

        self.approximator = approximator
        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_std = mu_prior_std
        self.num_dims = num_dims
        self.num_obs = num_obs
        self.likelihood_std = likelihood_std
        self.num_samples = num_samples
        self.model = assumed_model
        self.df = df
        self.use_student_t = use_student_t
        self.rng = rng if rng is not None else np.random.default_rng()
        self.logml_method = logml_method

    def normal_analytical(self, obs_data):
        for i in range(len(obs_data)):
            dataset = obs_data[i]["x"]
            analytical = GA(
                obs_data=dataset,
                num_dims=self.num_dims,
                mu_prior_mean=self.mu_prior_mean,
                mu_prior_std=self.mu_prior_std,
                num_obs=self.num_obs,
                likelihood_std=self.likelihood_std,
                num_samples=self.num_samples,
                rng=self.rng,
            )
            analytical_posterior_samples = analytical.analytical_posterior()
            log_marginal_analytical = analytical.log_marginal_analytical()
            obs_data[i][f"gold_log_marginal_{self.model}"] = log_marginal_analytical
            obs_data[i][f"gold_post_samples_{self.model}"] = (
                analytical_posterior_samples
            )
        return obs_data

    def npe_estimation(self, obs_data, seed: int | None = None):
        x_batch = np.stack([d["x"] for d in obs_data], axis=0)
        sample_kwargs = {
            "conditions": {"x": x_batch},
            "num_samples": self.num_samples,
        }
        if seed is not None:
            sample_kwargs["seed"] = int(seed)
        mu_samples = self.approximator.sample(**sample_kwargs)
        all_post_samples = np.asarray(mu_samples["mu"], dtype=np.float32)
        for i in range(len(obs_data)):
            dataset = obs_data[i]["x"]
            npe_post_samples = all_post_samples[i]
            estimator = MLE(
                approximator=self.approximator,
                mu=npe_post_samples,
                obs_data=dataset,
                mu_prior_mean=self.mu_prior_mean,
                mu_prior_std=self.mu_prior_std,
                num_dims=self.num_dims,
                likelihood_std=self.likelihood_std,
                df=self.df,
                use_student_t=self.use_student_t,
                rng=self.rng,
            )
            npe_log_marginal = estimator.log_marginal_npe(method=self.logml_method)
            obs_data[i][f"npe_post_samples_{self.model}"] = npe_post_samples
            obs_data[i][f"npe_log_marginal_{self.model}"] = npe_log_marginal
            obs_data[i][f"importance_ess_{self.model}"] = estimator.importance_ess
            obs_data[i][f"num_npe_logml_draws_{self.model}"] = (
                estimator.num_importance_samples
            )
        return obs_data

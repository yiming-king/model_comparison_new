functions {
    real wald_lpdf(real y, real alpha, real nu) {
        real lpdf;

        lpdf = (
            log(alpha) - 1.0/2.0 * log(2*pi()) - 3.0/2.0*log(y) - (alpha - nu*y)^2/(2*y)
        );

        return lpdf;
    }

    real wald_lcdf(real y, real alpha, real nu) {
        real mu = alpha/nu;
        real lambda = alpha^2;

        real ly = sqrt(lambda/y);
        real ymu = y/mu;

        vector[2] terms;

        terms[1] = std_normal_lcdf(ly * (ymu - 1));
        terms[2] = 2*lambda/mu;
        terms[2] += std_normal_lcdf(- ly * (ymu + 1));

        real result = log_sum_exp(terms);

        return result;
    }

    real wald_lccdf(real y, real alpha, real nu) {
        real lcdf = wald_lcdf(y | alpha, nu);
        return log1m_exp(lcdf);
    }

    real wald_rng(real alpha, real nu) {
        real mu = alpha/nu;
        real lambda = alpha^2;
        real zeta = normal_rng(0, 1);
        real zeta_sq = zeta^2;
        real x = mu + (mu^2*zeta_sq)/(2*lambda) - mu/(2*lambda)*sqrt(4*mu*lambda*zeta_sq + mu^2*zeta_sq^2);
        real z = uniform_rng(0, 1);
        real y;

        if(z <= mu / (mu + x)){
            y = x;
        } else {
            y = mu^2/x;
        }

        return y;
    }

    real rdm_lpdf(real rt, array[] real alpha, array[] real nu, real t0) {
        real lpdf = 0.0;
        real lccdf = 0.0;

        if (rt >= 0.0) {
          lpdf  = wald_lpdf (rt - t0 | alpha[1], nu[1]);
          lccdf = wald_lccdf(rt - t0 | alpha[2], nu[2]);
        } else {
          lpdf  = wald_lpdf (abs(rt) - t0 | alpha[2], nu[2]);
          lccdf = wald_lccdf(abs(rt) - t0 | alpha[1], nu[1]);
        }

        return lpdf + lccdf;
    }
}
data {
  int<lower=0> N;
  int<lower=0, upper=3> model_index; // 0 = null model, 1 = threshold by condition, 2 = threshold by accumulator, 3 = threshold by condition and accumulator
  array[N] real rt; // response times; negative values indicate an incorrect response
  array[N] int<lower=0, upper=1> condition; // experimental conditions
}
transformed data {
  int n_alpha; // number of threshold parameters
  array[N] real<lower=0> abs_rt = abs(rt);
  real min_rt = min(abs_rt);

  if (model_index == 0) {
    n_alpha = 1;
  } else if (model_index == 1 || model_index == 2) {
    n_alpha = 2;
  } else if (model_index == 3) {
    n_alpha = 4;
  }
}
parameters {
   array[n_alpha] real log_alpha;
   array[2] real log_nu;
   real logit_tau;
}
transformed parameters {
    array[N] real log_lik;
    array[2, 2] real alpha; //[2 x condition, 2 x accumulator]
    real total_log_lik = 0.0;
    array[2] real nu = exp(log_nu);
    real tau = inv_logit(logit_tau);
    real t0 = tau * min_rt;

    if (model_index == 0) {
        alpha[1, 1] = exp(log_alpha[1]);
        alpha[1, 2] = exp(log_alpha[1]);
        alpha[2, 1] = exp(log_alpha[1]);
        alpha[2, 2] = exp(log_alpha[1]);
    } else if (model_index == 1) {
        alpha[1, 1] = exp(log_alpha[1]);
        alpha[1, 2] = exp(log_alpha[1]);
        alpha[2, 1] = exp(log_alpha[2]);
        alpha[2, 2] = exp(log_alpha[2]);
    } else if (model_index == 2) {
        alpha[1, 1] = exp(log_alpha[1]);
        alpha[1, 2] = exp(log_alpha[2]);
        alpha[2, 1] = exp(log_alpha[1]);
        alpha[2, 2] = exp(log_alpha[2]);
    } else if (model_index == 3) {
        alpha[1, 1] = exp(log_alpha[1]);
        alpha[1, 2] = exp(log_alpha[2]);
        alpha[2, 1] = exp(log_alpha[3]);
        alpha[2, 2] = exp(log_alpha[4]);
    }

    for (n in 1:N) {
        log_lik[n] = rdm_lpdf(rt[n] | alpha[condition[n] + 1], nu, t0);
        total_log_lik += log_lik[n];
    }
}
model {
    target += normal_lpdf(log_alpha | 0.0, 0.5);
    target += normal_lpdf(log_nu    | 0.0, 0.5);
    target += normal_lpdf(logit_tau | 0.0, 1.0);
    
    target += total_log_lik;
    target += log1m(tau);
}

import keras.ops as ops
import keras.utils as utils
import tensorflow as tf
from math import log, pi
from bayesflow.types import Tensor
from .normal import normal_lcdf
from .utils import log1m_exp

@utils.register_keras_serializable("bayesflow.utils")
def wald_lpdf(x: Tensor, alpha: Tensor, nu: Tensor) -> Tensor:
    lpdf = (
        ops.log(alpha)
        - ops.convert_to_tensor(0.5 * log(2 * pi), dtype=ops.dtype(x))
        - 1.5 * ops.log(x)
        - ops.square(alpha - nu * x) / (2 * x)
    )

    return lpdf

@utils.register_keras_serializable("bayesflow.utils")
def wald_lcdf(x: Tensor, alpha: Tensor, nu: Tensor) -> Tensor:
    dtype, z, second = _wald_cdf_terms(x, alpha, nu)
    value = tf.reduce_logsumexp(tf.stack([normal_lcdf(z), second]), axis=0)
    return tf.cast(value, dtype)

def _wald_cdf_terms(x, alpha, nu):
    # Accumulate the cancellation-prone second term in double precision, even
    # for float32 callers. This changes arithmetic precision, not the Wald law.
    x = tf.convert_to_tensor(x)
    dtype = x.dtype
    x, alpha, nu = (tf.cast(v, tf.float64) for v in (x, alpha, nu))
    root = tf.sqrt(x)
    z = (nu * x - alpha) / root
    second = 2.0 * alpha * nu + normal_lcdf(-(nu * x + alpha) / root)
    return dtype, z, second

@utils.register_keras_serializable("bayesflow.utils")
def wald_lccdf(x: Tensor, alpha: Tensor, nu: Tensor) -> Tensor:
    # Survival = Phi(-z) - exp(2*alpha*nu)*Phi(-(nu*x+alpha)/sqrt(x)).
    # Subtract on the log scale instead of subtracting a rounded CDF from one.
    dtype, z, second = _wald_cdf_terms(x, alpha, nu)
    first = normal_lcdf(-z)
    value = first + log1m_exp(second - first)
    return tf.cast(value, dtype)

@utils.register_keras_serializable("bayesflow.utils")
def rdm_lpdf(rt: Tensor, alpha: Tensor, nu: Tensor, tau: Tensor) -> Tensor:
    # rt.shape = (batch_size, num_trials)
    # alpha.shape = (batch_size, num_trials, 2)
    # nu.shape = (batch_size, 1, 2)
    # tau.shape = (batch_size, 1)

    # non-decision time
    t0 = tau * ops.min(ops.abs(rt), axis=-1, keepdims=True)
    # decision time
    t = ops.abs(rt) - t0
    valid_support = t > 0
    safe_t = ops.where(valid_support, t, ops.ones_like(t))

    lpdf = ops.where(
        rt > 0,
        wald_lpdf(x=safe_t, alpha=alpha[..., 0], nu=nu[..., 0]),
        wald_lpdf(x=safe_t, alpha=alpha[..., 1], nu=nu[..., 1]),
    )

    lccdf = ops.where(
        rt < 0,
        wald_lccdf(x=safe_t, alpha=alpha[..., 0], nu=nu[..., 0]),
        wald_lccdf(x=safe_t, alpha=alpha[..., 1], nu=nu[..., 1]),
    )

    result = lpdf + lccdf

    # Outside the support the density is zero. NaN/+inf inside the support
    # remain visible to the estimator, which must reject numerical failures.
    return ops.where(valid_support, result, -float("inf"))

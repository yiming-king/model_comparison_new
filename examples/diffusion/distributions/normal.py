import keras.ops as ops
import keras.utils as utils
import tensorflow as tf
from bayesflow.types import Tensor
from math import log, pi


@utils.register_keras_serializable("bayesflow.utils")
def normal_lpdf(x: Tensor, mu: float = 0.0, sigma: float = 1.0) -> Tensor:
    lpdf = -0.5 * ops.square(x - mu) / ops.square(sigma)

    lpdf -= 0.5 * ops.log(2 * pi)
    lpdf -= ops.log(sigma)  # 0.5 log(sigma^2)
    return lpdf





@utils.register_keras_serializable("bayesflow.utils")
def normal_lcdf(
    x: Tensor,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> Tensor:
    """Compute a stable Normal log-CDF with the TensorFlow backend.

    Supports float32 and float64. Requires sigma > 0.
    """
    x = tf.convert_to_tensor(x)

    if x.dtype not in (tf.float32, tf.float64):
        raise TypeError("normal_lcdf requires float32 or float64 input.")

    mu = tf.cast(mu, x.dtype)
    sigma = tf.cast(sigma, x.dtype)
    z = (x - mu) / sigma

    inv_sqrt2 = tf.constant(2.0 ** -0.5, dtype=x.dtype)
    log2 = tf.constant(log(2.0), dtype=x.dtype)
    half_log2pi = tf.constant(0.5 * log(2.0 * pi), dtype=x.dtype)

    cutoff = -20.0 if x.dtype == tf.float64 else -10.0

    # Moderate negative inputs: avoid cancellation in 1 + erf(...).
    z_middle = tf.clip_by_value(z, cutoff, 0.0)
    log_middle = tf.math.log(
        tf.math.erfc(-z_middle * inv_sqrt2)
    ) - log2

    # Positive inputs: preserve small negative log-CDF values.
    z_positive = tf.maximum(z, 0.0)
    log_positive = tf.math.log1p(
        -0.5 * tf.math.erfc(z_positive * inv_sqrt2)
    )

    # Far negative tail: compute directly on the logarithmic scale.
    z_tail = tf.minimum(z, cutoff)
    r = tf.math.reciprocal(tf.square(z_tail))

    correction = r * (
        -1.0 + r * (
            3.0 + r * (
                -15.0 + r * (
                    105.0 + r * (-945.0 + 10395.0 * r)
                )
            )
        )
    )

    log_tail = (
        -0.5 * tf.square(z_tail)
        - tf.math.log(-z_tail)
        - half_log2pi
        + tf.math.log1p(correction)
    )

    return tf.where(
        z < cutoff,
        log_tail,
        tf.where(z > 0.0, log_positive, log_middle),
    )

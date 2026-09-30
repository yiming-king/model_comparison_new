import keras.ops as ops
import keras.utils as utils
from bayesflow.types import Tensor


@utils.register_keras_serializable("bayesflow.utils")
def expit(x: Tensor) -> Tensor:
    return 1 / (1 + ops.exp(-x))


@utils.register_keras_serializable("bayesflow.utils")
def log_sum_exp(x: Tensor, axis=-1) -> Tensor:
    a = ops.max(x, axis=axis, keepdims=True)
    sum_exp = ops.sum(ops.exp(x - a), axis=axis, keepdims=False)
    a = ops.squeeze(a, axis=axis)
    return a + ops.log(sum_exp)


@utils.register_keras_serializable("bayesflow.utils")
def log1m_exp(x):
    """Compute log(1 - exp(x)) for x < 0."""
    log_half = ops.log(0.5)
    log_half = ops.cast(log_half, ops.dtype(x))

    return ops.where(
        x < log_half,
        ops.log1p(-ops.exp(x)),
        ops.log(-ops.expm1(x)),
    )

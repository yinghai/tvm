"""TVM operator batch matmul compute."""
from __future__ import absolute_import
import tvm
from .. import tag


def batch_matmul_default(A, B):
    """The default implementation of batched_matmul in topi.

    Parameters
    ----------
    A: tvm.Tensor
        3-D with shape [batch, N, K]

    B: tvm.Tensor
        3-D with shape [batch, K, M]

    Returns
    -------
    output: tvm.Tensor
        3-D with shape [batch, N, M]
    """
    assert len(A.shape) == 3 and len(B.shape) == 3, \
        "only support 3-dim batch matmul"
    batch, N, K = A.shape
    _, _, M = B.shape
    k = tvm.reduce_axis((0, K), name='k')

    bmm = tvm.compute((batch, N, M),
                      lambda b, y, x: tvm.sum(A[b, y, k] * B[b, k, x], axis=k),
                      tag='batch_matmul')
    return bmm

@tvm.target.override_native_generic_func("batch_matmul")
def batch_matmul(A, B):
    """Applies batched matmul.

    Parameters
    ----------
    A: tvm.Tensor
        3-D with shape [batch, N, K]

    B: tvm.Tensor
        3-D with shape [batch, K, M]

    Returns
    -------
    output: tvm.Tensor
        3-D with shape [batch, N, M]
    """
    return batch_matmul_default(A, B)

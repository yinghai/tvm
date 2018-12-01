"""TVM operator batch matmul compute."""
from __future__ import absolute_import
import topi
import tvm
from tvm import autotvm
from tvm.autotvm.task.nnvm_integration import deserialize_args
from tvm.contrib import cblas
from .. import generic, tag
from ..nn.batch_matmul import batch_matmul
from ..util import traverse_inline, get_const_tuple

@autotvm.register_topi_compute(batch_matmul, 'cpu', 'direct')
def _declaration_batch_matmul(cfg, A, B, trans_a, trans_b):
    batch_a = A.shape[0]
    batch_b = B.shape[0]
    N = A.shape[-1] if trans_a else A.shape[-2]
    K = A.shape[-2] if trans_a else A.shape[-1]
    M = B.shape[-2] if trans_b else B.shape[-1]
    oshape = A.shape[:-2] + [N, M]
    ndim = len(A.shape)
    if ndim > 3:
        batch_a = product(A.shape[:-2])
        batch_b = product(B.shape[:-2])
        A = topi.reshape(A, [batch_a] + A.shape[-2:])
        B = topi.reshape(B, [batch_b] + B.shape[-2:])

    if cfg['use_cblas'].val:
        C = cblas.batch_matmul(A, B, trans_a, trans_b)
    else:
        if trans_a:
            A = topi.transpose(A, [0, 2, 1])
        if trans_b:
            B = topi.transpose(B, [0, 2, 1])

        k = tvm.reduce_axis((0, K), name='k')
        C = tvm.compute((batch_a, N, M),
                         lambda b, y, x: tvm.sum(A[b, y, k] * B[b, k, x], axis=k),
                         tag='batch_matmul')

    if ndim > 3:
        return topi.reshape(C, oshape)
    else:
        return C


def _schedule_batch_matmul_impl(cfg, s, output, C):
    bb, x, y = s[C].op.axis
    k, = s[C].op.reduce_axis

    ##### define space begin #####
    cfg.define_knob('split_x', [16, 32, 64, 128, 256])
    cfg.define_knob('split_y', [16, 32, 64, 128, 256])
    cfg.define_knob('split_k', [4, 8, 16, 32, 64])
    ##### define space end #####

    # schedule according to config
    xo, xi = s[C].split(x, factor=cfg['split_x'].val)
    yo, yi = s[C].split(y, factor=cfg['split_y'].val)
    ko, ki = s[C].split(k, factor=cfg['split_k'].val)

    s[C].reorder(xo, yo, ko, xi, ki, yi)

    s[C].unroll(ki)
    s[C].vectorize(yi)


@autotvm.register_topi_schedule(generic.schedule_batch_matmul, 'cpu', ['direct'])
def _schedule_batch_matmul(cfg, outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def _callback(op):
        if 'batch_matmul' in op.tag:
            _schedule_batch_matmul_impl(cfg, s, op.output(0), outs[0])

    if not cfg['use_cblas'].val:
        traverse_inline(s, outs[0].op, _callback)
    return s

# Define template function for autotvm task
# We define schedule template in this function instead of
# declaration function since actual input arguments need
# to be altered by the schedule selected.
@autotvm.task.register("topi_x86_batch_matmul")
def _topi_x86_batch_matmul(*args, **kwargs):
    assert not kwargs, "Do not support kwargs in template function call"
    Batch, M, N, K, trans_a, trans_b = deserialize_args(args)

    cfg = autotvm.get_config()
    cfg.define_knob('use_cblas', [0, 1])

    a_shape = (Batch, M, K) if not trans_a else (Batch, K, M)
    b_shape = (Batch, K, N) if not trans_b else (Batch, N, K)
    A = tvm.placeholder(a_shape, dtype='float32')
    B = tvm.placeholder(b_shape, dtype='float32')
    C = _declaration_batch_matmul(cfg, A, B, trans_a, trans_b)
    s = _schedule_batch_matmul(cfg, [C])
    return s, [A, B, C]

# pylint: disable=invalid-name,unused-variable
"""dense schedule on ARM Mali GPU"""

from __future__ import absolute_import as _abs

import tvm
from tvm import autotvm
import copy
from .. import generic, nn, tag
from ..util import traverse_inline, get_const_tuple
import numpy as np

from tvm.contrib import cblas
from .check_targets import fp32_vector_width

(TVM, BLAS, BLAS_PRETRANSPOSED) = range(3)


@autotvm.register_topi_compute(nn.dense, 'cpu', ['direct'])
def dense(cfg, data, weight, bias=None):
    """The default implementation of dense in topi.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1

    batch, in_dim = get_const_tuple(data.shape)
    out_dim, _ = get_const_tuple(weight.shape)

    cfg.define_knob('blas', [TVM, BLAS, BLAS_PRETRANSPOSED])
    f = [dense_direct, dense_blas, dense_blas_pretranspose][cfg['blas'].val]
    matmul = f(cfg, data, weight)
    if bias is not None:
        matmul = tvm.compute(
            (batch, out_dim),
            lambda i, j: matmul[i, j] + bias[j],
            name="bias_add",
            tag=tag.BROADCAST
        )
    cfg.add_flop(2 * batch * in_dim * out_dim)
    if bias is not None:
        cfg.add_flop(batch * out_dim)
    return matmul

@autotvm.register_topi_compute(nn.dense, 'cpu', ['pretransposed'])
def dense(cfg, data, weight, bias=None):
    """The default implementation of dense in topi.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1

    batch, in_dim = get_const_tuple(data.shape)
    _, out_dim = get_const_tuple(weight.shape)
    return cblas.matmul(data, weight, transb=False, tag="dense")

def dense_direct(cfg, data, weight):
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = tvm.reduce_axis((0, in_dim), name='k')
    return tvm.compute(
        (batch, out_dim),
        lambda i, j: tvm.sum(data[i, k] * weight[j, k], axis=k),
        tag='dense'
    )


def dense_blas(cfg, data, weight):
    return cblas.matmul(data, weight, transb=True, tag="dense")


def dense_blas_pretranspose(cfg, data, weight):
    import topi
    weight_T = topi.transpose(weight, [1, 0])
    return cblas.matmul(data, weight_T, transb=False, tag="dense")


@autotvm.register_topi_schedule(generic.schedule_dense, 'cpu', ['direct', 'pretransposed'])
def schedule_dense(cfg, outs):
    """Schedule for dense operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The config entity for this template
    outs: Array of Tensor
        The computation graph description of dense
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for dense.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'dense':
            data, weight = op.input_tensors
            if autotvm.GLOBAL_SCOPE.in_tuning and cfg['blas'].val == BLAS_PRETRANSPOSED:
                s[weight].pragma(s[weight].op.axis[0], "debug_skip_region")

    # TODO: autotune this?
    (_, CO) = get_const_tuple(outs[0].shape)
    if CO != 1:
        assert isinstance(
            outs[0].op, tvm.tensor.ComputeOp) or isinstance(
            outs[0].op, tvm.tensor.ExternOp)
        if isinstance(outs[0].op, tvm.tensor.ComputeOp):
            (co, ci) = s[outs[0].op].split(s[outs[0].op].op.axis[1],
                                           fp32_vector_width())
            s[outs[0].op].vectorize(ci)
    traverse_inline(s, outs[0].op, _callback)

    return s

@nn.dense_alter_layout.register("cpu")
def dense_alter_layout(attrs, inputs, tinfo):
    import nnvm.symbol as sym
    dispatch_ctx = autotvm.task.DispatchContext.current
    target = tvm.target.current_target()
    # query schedule and fallback if necessary
    workload = autotvm.task.args_to_workload(tinfo, nn.dense)
    cfg = dispatch_ctx.query(target, workload)
    if cfg.is_fallback:
        return None
    if cfg['blas'].val != BLAS_PRETRANSPOSED:
        return None

    weights = tinfo[1]
    transposed_weights = sym.transpose(
        inputs[1],
        axes=(1, 0),
        # to work around unique name check in PrecomputePrune
        name="weight_transpose_{}".format(np.random.randint(0, 1E8))
    )
    transposed_weights_placeholder = tvm.placeholder((weights.shape[1], weights.shape[0]),
                                              dtype=weights.dtype)
    transposed_workload = autotvm.task.args_to_workload(
        [tinfo[0], transposed_weights_placeholder, tinfo[2]], nn.dense)
    transposed_inputs = [inputs[0], transposed_weights, inputs[2]]
    transposed_cfg = copy.deepcopy(cfg)
    transposed_cfg.template_key = "pretransposed"
    dispatch_ctx.update(target, transposed_workload, transposed_cfg)

    ret = sym.dense(*transposed_inputs, **attrs)

    return ret

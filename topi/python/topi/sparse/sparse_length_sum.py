from ..util import get_const_int, get_const_tuple
import tvm


def product(l):
    p = 1
    for i in l:
        p *= i
    return p


def cumsum(X):
    """
    Y[i] = sum(X[:i])
    """
    (m, ) = get_const_tuple(X.shape)
    s_state = tvm.placeholder((m + 1, ), dtype="int32", name="state")
    s_init = tvm.compute((1, ), lambda _: tvm.const(0, "int32"))
    s_update = tvm.compute((m + 1, ), lambda l: s_state[l - 1] + X[l - 1])
    return tvm.scan(s_init, s_update, s_state, inputs=[X], name="cumsum")


def sparse_length_sum(data, indices, lengths):
    G = get_const_int(lengths.shape[0])
    D = product(get_const_tuple(data.shape)[1:])
    Dloop = "vectorize" if D > 1 else "serial"
    def sparse_length_sum_ir(data, indices, lengths, lengths_offsets, out):
        irb = tvm.ir_builder.create()
        data_ptr = irb.buffer_ptr(data)
        indices_ptr = irb.buffer_ptr(indices)
        lengths_ptr = irb.buffer_ptr(lengths)
        lengths_offsets_ptr = irb.buffer_ptr(lengths_offsets)

        data_stride = data.strides[0] if data.strides else data.shape[1]
        out_stride = out.strides[0] if out.strides else out.shape[1]
        out_ptr = irb.buffer_ptr(out)
        with irb.for_range(0, G, name='g') as g:
            acc = irb.allocate('float32', (D, ), name='dot', scope='local')
            with irb.for_range(0, D, for_type=Dloop, name='d') as d:
                acc[d] = tvm.const(0, "float32")
            with irb.for_range(0, lengths_ptr[g], name='gg') as gg:
                offset = gg + lengths_offsets_ptr[g]
                with irb.for_range(0, D, for_type=Dloop, name='d') as d:
                    acc[d] += data_ptr[indices_ptr[offset] * data_stride + d]
            with irb.for_range(0, D, for_type=Dloop, name='d') as d:
                out_ptr[g * out_stride + d] = acc[d]
        return irb.get()

    oshape = list(get_const_tuple(data.shape))
    oshape[0] = get_const_int(lengths.shape[0])
    length_offsets = cumsum(lengths)
    sparse_length_sum = tvm.extern(
        oshape, [data, indices, lengths, length_offsets],
        lambda ins, outs: sparse_length_sum_ir(ins[0], ins[1], ins[2], ins[3], outs[0]),
        tag="sparse_length_sum",
        dtype="float32",
        name="sparse_length_sum")
    return sparse_length_sum


def sparse_length_sum_fused_8_bit_rowwise(data, indices, lengths):
    assert data.dtype == "uint8"
    G = get_const_int(lengths.shape[0])
    # This uses a bizarre representation, where we pack a [4xuint8_t scale,
    # 4xuint8_t bias, uint8_t* data].
    D = product(get_const_tuple(data.shape)[1:]) - 8
    assert D > 0
    Dloop = "vectorize" if D > 1 else "serial"

    def sparse_length_sum_fused_8_bit_rowwise_ir(data, indices, lengths,
                                                 lengths_offsets, out):
        irb = tvm.ir_builder.create()
        data_ptr = irb.buffer_ptr(data)
        indices_ptr = irb.buffer_ptr(indices)
        lengths_ptr = irb.buffer_ptr(lengths)
        lengths_offsets_ptr = irb.buffer_ptr(lengths_offsets)

        data_stride = data.strides[0] if data.strides else data.shape[1]
        out_stride = out.strides[0] if out.strides else out.shape[1]
        out_ptr = irb.buffer_ptr(out)
        with irb.for_range(0, G, name='g') as g:
            acc = irb.allocate('float32', (D, ), name='dot', scope='local')

            with irb.for_range(0, D, for_type=Dloop, name='d') as d:
                acc[d] = tvm.const(0, "float32")
            with irb.for_range(0, lengths_ptr[g], name='gg') as gg:

                scale = irb.allocate('float32', (1, ), name='scale', scope='local')
                bias = irb.allocate('float32', (1, ), name='bias', scope='local')

                offset = gg + lengths_offsets_ptr[g]
                data_index = indices_ptr[offset]
                scale_uint8 = data.vload([data_index, 0], "uint8x4")
                bias_uint8 = data.vload([data_index, 4], "uint8x4")
                scale[0] = tvm.call_pure_intrin('float32', 'reinterpret', scale_uint8)
                bias[0] = tvm.call_pure_intrin('float32', 'reinterpret', bias_uint8)
                with irb.for_range(0, D, for_type=Dloop, name='d') as d:
                    d_uint8 = data_ptr[data_index * data_stride + d + 8]
                    d_float32 = tvm.make.static_cast("float32", d_uint8) * scale[0] + bias[0]
                    acc[d] += d_float32
            with irb.for_range(0, D, for_type=Dloop, name='d') as d:
                out_ptr[g * out_stride + d] = acc[d]
        return irb.get()

    oshape = (get_const_int(lengths.shape[0]), D)

    length_offsets = cumsum(lengths)
    sparse_length_sum = tvm.extern(
        oshape, [data, indices, lengths, length_offsets],
        lambda ins, outs: sparse_length_sum_fused_8_bit_rowwise_ir(ins[0], ins[1], ins[2], ins[3], outs[0]),
        tag="sparse_length_sum",
        dtype="float32",
        name="sparse_length_sum")
    return sparse_length_sum

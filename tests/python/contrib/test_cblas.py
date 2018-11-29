import tvm
import numpy as np
from tvm.contrib import cblas
from topi.util import get_const_tuple

def test_matmul_add():
    n = 1024
    l = 128
    m = 235
    bias = tvm.var('bias', dtype=tvm.float32)
    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((l, m), name='B')
    C = cblas.matmul(A, B)
    D = tvm.compute(C.shape, lambda i, j: C[i,j] + bias, name="D")
    s = tvm.create_schedule(D.op)

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.cblas.matmul", True):
            print("skip because extern function is not available")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [A, B, D, bias], target)
        a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(l, m)).astype(B.dtype), ctx)
        d = tvm.nd.array(np.zeros((n, m), dtype=D.dtype), ctx)
        bb = 10.0
        f(a, b, d, bb)
        tvm.testing.assert_allclose(
            d.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()) + bb, rtol=1e-5)
    verify()


def test_batch_matmul(batch, n, l, m, trans_a=False, trans_b=False):
    dtype = 'float32'
    A = tvm.placeholder((batch, l if trans_a else n, n if trans_a else l), name='A')
    B = tvm.placeholder((batch, m if trans_b else l, l if trans_b else m), name='B')
    C = cblas.batch_matmul(A, B, trans_a, trans_b)
    s = tvm.create_schedule(C.op)

    def get_ref_data():
        a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(dtype)
        b_np = np.random.uniform(size=get_const_tuple(B.shape)).astype(dtype)
        aa_np = a_np.transpose([0, 2, 1]) if trans_a else a_np
        bb_np = b_np.transpose([0, 2, 1]) if trans_b else b_np
        c_np = np.random.uniform(size=(batch, n, m)).astype(dtype)
        for i in range(batch):
            c_np[i, :, :] = np.dot(aa_np[i, :, :], bb_np[i, :, :])
        return a_np, b_np, c_np
    # get the test data
    a_np, b_np, c_np = get_ref_data()

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.cblas.batch_matmul", True):
            print("skip because extern function is not available")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [A, B, C], target)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros((batch, n, m), dtype=dtype), ctx)
        f(a, b, c)
        tvm.testing.assert_allclose(
            c.asnumpy(), c_np, rtol=1e-5)
    verify()

if __name__ == "__main__":
    test_matmul_add()
    test_batch_matmul(4, 2, 5, 3)
    test_batch_matmul(4, 2, 5, 3, trans_a=True)
    test_batch_matmul(4, 2, 5, 3, trans_b=True)
    test_batch_matmul(4, 2, 5, 3, trans_a=True, trans_b=True)

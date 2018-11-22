"""Test code for batch_matmul operator"""
import numpy as np
import tvm
import topi
import topi.testing
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize

from common import get_all_backend

def verify_batch_matmul(batch, N, K, M):
    A = tvm.placeholder((batch, N, K), name='A')
    B = tvm.placeholder((batch, K, M), name='B')
    C = tvm.placeholder((batch, N, M), name='C')
    dtype = A.dtype

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_batch_matmul")
    def get_ref_data():
        a_np = np.random.uniform(size=(batch, N, K)).astype(dtype)
        b_np = np.random.uniform(size=(batch, K, M)).astype(dtype)
        c_np = np.random.uniform(size=(batch, N, M)).astype(dtype)
        for i in range(batch):
            c_np[i, :, :] = np.matmul(a_np[i, :, :], b_np[i, :, :])
        return (a_np, b_np, c_np)
    # get the test data
    a_np, b_np, c_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            C = topi.nn.batch_matmul(A, B)
            s = topi.generic.schedule_dense(C)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B, C], device, name="batch_matmul")
        f(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    for device in get_all_backend():
        check_device(device)

def test_batch_matmul():
    verify_batch_matmul(4, 1024, 1000, 256)

if __name__ == "__main__":
    test_batch_matmul()

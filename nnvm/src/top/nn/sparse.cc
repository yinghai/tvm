
#include "topi/nn.h"
#include "topi/nn/dense.h"
#include "topi/nn/softmax.h"
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/compiler/util.h>
#include <nnvm/top/nn.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/layout.h>
#include <nnvm/node.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/nn.h>
#include <tvm/expr.h>
#include <tvm/packed_func_ext.h>
#include <tvm/tvm.h>

#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace nnvm {
namespace top {

using tvm::Var;
using tvm::Expr;
using tvm::Tensor;
using tvm::Array;
using nnvm::compiler::FTVMCompute;

inline bool SparseLengthSumInferShape(const nnvm::NodeAttrs &attrs,
                                      std::vector<TShape> *in_shape,
                                      std::vector<TShape> *out_shape) {
  CHECK_EQ(in_shape->size(), 3);
  CHECK_EQ(out_shape->size(), 1);
  TShape dshape = (*in_shape)[0];
  TShape lshape = (*in_shape)[2];
  CHECK_EQ(dshape.ndim(), 2);
  TShape oshape = dshape;
  oshape[0] = lshape[0];
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(_contrib_sparse_lengths_sum)
    .describe(R"code(Applies the sparse length sum transformation.

- **data**: `(M, K)`
- **indices**: `(sum(lengths),)`
- **lengths**: `(L,)`
- **out**: `(L, K)`

The reference implementation (in Python) is simply:

    def sparse_length_sum_ref(D, I, L):
        R = np.zeros(shape=(L.size, ) + D.shape[1:], dtype=D.dtype)
        Lsum = np.cumsum([0] + L.tolist())
        for g in range(L.size):
            for gg in range(L[g]):
                R[g, :] += D[I[Lsum[g] + gg], :]
        return R

)code" NNVM_ADD_FILELINE)
    .add_argument("data", "2D Tensor", "Input data.")
    .add_argument("indices", "1D Tensor", "Indices for lookups.")
    .add_argument("lengths", "1D Tensor", "Segment lengths.")
    .set_attr_parser(ParamParser<DenseParam>)
    .set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<DenseParam>)
    .set_num_outputs(1)
    .set_num_inputs(3)
    .set_attr<FInferShape>("FInferShape", SparseLengthSumInferShape)
    .set_attr<FInferType>("FInferType", ElemwiseType<3, 1>)
    .set_support_level(1);

} // namespace top
} // namespace nnvm

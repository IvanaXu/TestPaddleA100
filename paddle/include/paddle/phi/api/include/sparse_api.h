#pragma once

#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace experimental {
namespace sparse {


// out, rulebook

PADDLE_API Tensor conv3d(const Tensor& x, const Tensor& kernel, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides, int groups, bool subm);


// out

PADDLE_API Tensor coo_to_dense(const Tensor& x);


// out

PADDLE_API Tensor create_sparse_coo_tensor(const Tensor& values, const Tensor& indices, const IntArray& dense_shape);


// out

PADDLE_API Tensor dense_to_coo(const Tensor& x, int64_t sparse_dim);


// out

PADDLE_API Tensor relu(const Tensor& x);


// out@SparseCooTensor

PADDLE_API Tensor sin(const Tensor& x);


// out

PADDLE_API Tensor sqrt(const Tensor& x);


// out

PADDLE_API Tensor tanh(const Tensor& x);


// out

PADDLE_API Tensor to_dense(const Tensor& x);


// out

PADDLE_API Tensor to_sparse_coo(const Tensor& x, int64_t sparse_dim);


// out

PADDLE_API Tensor to_sparse_csr(const Tensor& x);


// out

PADDLE_API Tensor values(const Tensor& x);


// out, rulebook

PADDLE_API Tensor maxpool(const Tensor& x, const std::vector<int>& kernel_sizes, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides);



}  // namespace sparse
}  // namespace experimental
}  // namespace paddle

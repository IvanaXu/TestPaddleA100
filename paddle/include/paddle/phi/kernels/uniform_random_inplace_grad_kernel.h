/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void UniformRandomInplaceGradKernel(const Context& ctx,
                                    const DenseTensor& out_grad,
                                    float min,
                                    float max,
                                    int seed,
                                    int diag_num,
                                    int diag_step,
                                    float diag_val,
                                    DenseTensor* x_grad);

}  // namespace phi

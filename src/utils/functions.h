#pragma once

#include "../tensor/tensor.h"

namespace NSTTF {
void init();
Tensor sum(Tensor &arg1, Tensor &arg2);
Tensor subtraction(Tensor &arg1, Tensor &arg2);
Tensor callFunction(const std::string &name,
                    const std::vector<Tensor> &tensors);
void checkShape(Tensor &arg1, Tensor &arg2);
} // namespace NSTTF
#pragma once

#include "../tensor/tensor.h"

namespace NSTTF {
namespace functions {
void init();
}
Tensor sum(const std::vector<Tensor> &tensors);
Tensor subtraction(const std::vector<Tensor> &tensors);
Tensor multiplication(const std::vector<Tensor> &tensors);
Tensor matrix_multiplication(const std::vector<Tensor> &tensors);
Tensor matrix_transpose(const std::vector<Tensor> &tensors);

Tensor callFunction(std::string &name, const std::vector<Tensor> &tensors);

void checkNumOfTensors(const std::vector<Tensor> &tensors, size_t num);
void checkShape(Tensor &arg1, Tensor &arg2);
} // namespace NSTTF

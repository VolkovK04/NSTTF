#pragma once

#include "../tensor/tensor.h"

namespace NSTTF {
namespace functions {
void init();
}
Tensor sum(std::vector<Tensor> &tensors);
Tensor subtraction(std::vector<Tensor> &tensors);
Tensor multiplication(std::vector<Tensor> &tensors);
Tensor matrix_multiplication(std::vector<Tensor> &tensors);
Tensor matrix_transpose(std::vector<Tensor> &tensors);

Tensor callFunction(std::string &name, std::vector<Tensor> &tensors);

void checkNumOfTensors(const std::vector<Tensor> &tensors, size_t num);
void checkShape(Tensor &arg1, Tensor &arg2);
} // namespace NSTTF

#pragma once

#include "../tensor/tensor.h"

namespace NSTTF {
void init();
Tensor sum(Tensor &arg1, Tensor &arg2);
Tensor subtraction(Tensor &arg1, Tensor &arg2);

void checkShape(Tensor &arg1, Tensor &arg2);
} // namespace NSTTF
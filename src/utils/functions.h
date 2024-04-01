#pragma once

#include "../tensor/tensor.h"

namespace NSTTF {
    void init();
    Tensor sum(Tensor& arg1, Tensor& arg2);
}
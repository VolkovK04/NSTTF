#pragma once

// #include <cl_functions/cl_functions.h>
#include <libutils/misc.h>
#include <tensor/tensor.h>
#include <fstream>

namespace NSTTF
{
    namespace functions
    {
        extern ocl::Kernel subtraction;
        extern ocl::Kernel sum;
        extern ocl::Kernel multiplication;
        extern ocl::Kernel matrix_multiplication;
        extern ocl::Kernel matrix_transpose;
        void init();
    }
    Tensor sum(const std::vector<Tensor> &tensors);
    Tensor subtraction(const std::vector<Tensor> &tensors);
    Tensor multiplication(const std::vector<Tensor> &tensors);
    Tensor matrix_multiplication(const std::vector<Tensor> &tensors);
    Tensor matrix_transpose(const std::vector<Tensor> &tensors);

    Tensor callFunction(const std::string &name,
                        const std::vector<Tensor> &tensors);

    void checkNumOfTensors(const std::vector<Tensor> &tensors, size_t num);
    void checkShape(Tensor &arg1, Tensor &arg2);
    std::vector<char> clToCharVector(const std::string& clFilename);
} // namespace NSTTF

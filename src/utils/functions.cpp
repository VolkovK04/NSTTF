#include "functions.h"


#include <libutils/misc.h>

#include "../cl_build_headers/matrix_multiplication_cl.h"
#include "../cl_build_headers/matrix_transpose_cl.h"
#include "../cl_build_headers/multiplication_cl.h"
#include "../cl_build_headers/subtraction_cl.h"
#include "../cl_build_headers/sum_cl.h"

namespace NSTTF {

namespace functions {
ocl::Kernel subtraction(subtraction_kernel, subtraction_kernel_length,
                        "subtraction");
ocl::Kernel multiplication(multiplication_kernel, multiplication_kernel_length,
                           "multiplication");
ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum");
ocl::Kernel matrix_multiplication(matrix_multiplication_kernel,
                                  matrix_multiplication_kernel_length,
                                  "matrix_multiplication_updated");
ocl::Kernel matrix_transpose(matrix_transpose_kernel,
                             matrix_transpose_kernel_length,
                             "matrix_transpose");

unsigned int workGroupSize = 128;
unsigned int global_work_size =
    (n + workGroupSize - 1) / workGroupSize * workGroupSize;

void init() {
    subtraction.compile();
    sum.compile();
    multiplication.compile();
    matrix_multiplication.compile();
    matrix_transpose.compile();
}
} // namespace functions

Tensor sum(Tensor &arg1, Tensor &arg2) {
    if (arg1.getShape() != arg2.getShape()) {
        throw std::runtime_error("Different size");
    }
    Tensor res(arg1.getShape());
    // TODO: check to work
    functions::sum.exec(
        gpu::WorkSize(functions::workGroupSize, functions::global_work_size),
        arg1.getGPUBuffer(), arg2.getGPUBuffer(), res.getGPUBuffer());
    return res;
}
} // namespace NSTTF
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

unsigned int n = 128; // TODO: ХЗ чё за херь, надо поменять
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
    checkShape(arg1, arg2);

    Tensor res(arg1.getShape());
    functions::sum.exec(
        gpu::WorkSize(functions::workGroupSize, functions::global_work_size),
        arg1.getGPUBuffer(), arg2.getGPUBuffer(), res.getGPUBuffer(),
        // getSize(arg1.getShape()));
        arg1.getSize());

    return res;
}

Tensor subtraction(Tensor &arg1, Tensor &arg2) {
    checkShape(arg1, arg2);

    Tensor res(arg1.getShape());
    functions::subtraction.exec(
        gpu::WorkSize(functions::workGroupSize, functions::global_work_size),
        arg1.getGPUBuffer(), arg2.getGPUBuffer(), res.getGPUBuffer(),
        arg1.getSize());
    return res;
}

Tensor multiplication(Tensor &arg1, Tensor &arg2) {
    checkShape(arg1, arg2);

    Tensor res(arg1.getShape());
    functions::multiplication.exec(
        gpu::WorkSize(functions::workGroupSize, functions::global_work_size),
        arg1.getGPUBuffer(), arg2.getGPUBuffer(), res.getGPUBuffer(),
        arg1.getSize());
    return res;
}

// Пока считаем, что подаются только матрицы
Tensor matrix_multiplication(Tensor &arg1, Tensor &arg2) {
    std::vector<size_t> arg1Shape = arg1.getShape();
    std::vector<size_t> arg2Shape = arg2.getShape();

    if(arg1Shape[2] != arg2Shape[1]){
        throw std::runtime_error("Different size");
    }

    Tensor res(std::vector<size_t> {arg1Shape[0], arg1Shape[1], arg2Shape[2]});
    functions::matrix_multiplication.exec(
        gpu::WorkSize(functions::workGroupSize, functions::global_work_size),
        arg1.getGPUBuffer(), arg2.getGPUBuffer(), res.getGPUBuffer(),
        arg1Shape[1], arg2Shape[1], arg2Shape[2]);
    return res;
}

// Пока считаем, что подаются только матрицы
Tensor matrix_transpose(Tensor &arg) {
    std::vector<size_t> baseShape = arg.getShape();
    size_t rowCount = baseShape[1], columnCount = baseShape[2];

    Tensor res(std::vector<size_t> {baseShape[0], columnCount, rowCount});
    functions::matrix_transpose.exec(
        gpu::WorkSize(functions::workGroupSize, functions::global_work_size),
        arg.getGPUBuffer(), res.getGPUBuffer(), rowCount, columnCount);
    return res;
}

void checkShape(Tensor &arg1, Tensor &arg2) {
    if (arg1.getShape() != arg2.getShape()) {
        throw std::runtime_error("Different size");
    }
}

} // namespace NSTTF
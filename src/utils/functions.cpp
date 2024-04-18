#include "functions.h"

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

void init() {
    subtraction.compile();
    sum.compile();
    multiplication.compile();
    matrix_multiplication.compile();
    matrix_transpose.compile();
}
} // namespace functions

unsigned int workGroupSize = 128;

Tensor sum(const std::vector<Tensor> &tensors) {
    checkNumOfTensors(tensors, 2);

    Tensor arg1 = tensors[0];
    Tensor arg2 = tensors[1];

    checkShape(arg1, arg2);

    Tensor res(arg1.getShape());
    unsigned int n = arg1.getSize();
    unsigned int global_work_size =
        (n + workGroupSize - 1) / workGroupSize * workGroupSize;
    functions::sum.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        arg1.getGPUBuffer(), arg2.getGPUBuffer(),
                        res.getGPUBuffer(), n);

    return res;
}

Tensor subtraction(const std::vector<Tensor> &tensors) {
    checkNumOfTensors(tensors, 2);

    Tensor arg1 = tensors[0];
    Tensor arg2 = tensors[1];

    checkShape(arg1, arg2);

    Tensor res(arg1.getShape());
    unsigned int n = arg1.getSize();
    unsigned int global_work_size =
        (n + workGroupSize - 1) / workGroupSize * workGroupSize;
    functions::subtraction.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                arg1.getGPUBuffer(), arg2.getGPUBuffer(),
                                res.getGPUBuffer(), n);
    return res;
}

Tensor multiplication(const std::vector<Tensor> &tensors) {
    checkNumOfTensors(tensors, 2);

    Tensor arg1 = tensors[0];
    Tensor arg2 = tensors[1];

    checkShape(arg1, arg2);

    Tensor res(arg1.getShape());
    unsigned int n = arg1.getSize();
    unsigned int global_work_size =
        (n + workGroupSize - 1) / workGroupSize * workGroupSize;
    functions::multiplication.exec(
        gpu::WorkSize(workGroupSize, global_work_size), arg1.getGPUBuffer(),
        arg2.getGPUBuffer(), res.getGPUBuffer(), n);
    return res;
}

// Пока считаем, что подаются только матрицы
Tensor matrix_multiplication(const std::vector<Tensor> &tensors) {
    checkNumOfTensors(tensors, 2);

    Tensor arg1 = tensors[0];
    Tensor arg2 = tensors[1];

    std::vector<size_t> arg1Shape = arg1.getShape();
    std::vector<size_t> arg2Shape = arg2.getShape();

    size_t arg1Col = arg1Shape[0],
          arg1Rows = arg1Shape[1];

    size_t arg2Col = arg2Shape[0],
          arg2Rows = arg2Shape[1];


    if (arg1Rows != arg2Col) {
        throw std::runtime_error("Wrong matrix shape");
    }

    arg1Shape.pop_back();
    arg1Shape.push_back(arg2Rows);

    Tensor res(arg1Shape);
    unsigned int M = arg1Col;
    unsigned int K = arg2Col;
    unsigned int N = arg2Rows;
    unsigned int x_work_group_size = 16;
    unsigned int y_work_group_size = 4;
    unsigned int x_work_size =
        (M + x_work_group_size - 1) / x_work_group_size * x_work_group_size;
    unsigned int y_work_size =
        (N + y_work_group_size - 1) / y_work_group_size * y_work_group_size;
    functions::matrix_multiplication.exec(
        gpu::WorkSize(gpu::WorkSize(x_work_group_size, y_work_group_size,
                                    x_work_size, y_work_size)),
        arg1.getGPUBuffer(), arg2.getGPUBuffer(), res.getGPUBuffer(), M, K, N);
    return res;
}

// Пока считаем, что подаются только матрицы
Tensor matrix_transpose(const std::vector<Tensor> &tensors) {
    checkNumOfTensors(tensors, 1);

    Tensor arg = tensors[0];

    std::vector<size_t> baseShape = arg.getShape();

    size_t argCol = baseShape[1],
      argRows = baseShape[0];

    baseShape[0] = argCol;
    baseShape[1] = argRows;

    Tensor res(baseShape);
    unsigned int M = argRows;
    unsigned int K = argCol;
    unsigned int x_work_group_size = 16;
    unsigned int y_work_group_size = 16; // ???
    unsigned int x_work_size =
        (M + x_work_group_size - 1) / x_work_group_size * x_work_group_size;
    unsigned int y_work_size =
        (K + y_work_group_size - 1) / y_work_group_size * y_work_group_size;
    functions::matrix_transpose.exec(
        gpu::WorkSize(x_work_group_size, y_work_group_size, x_work_size,
                      y_work_size),
        arg.getGPUBuffer(), res.getGPUBuffer(), M, K);
    return res;
}

void checkShape(Tensor &arg1, Tensor &arg2) {
    if (arg1.getShape() != arg2.getShape()) {
        throw std::runtime_error("Different shape");
    }
}

void checkNumOfTensors(const std::vector<Tensor> &tensors, size_t num) {
    if (tensors.size() < num) {
        throw std::runtime_error("Not enought tensors");
    } else if (tensors.size() > num) {
        throw std::runtime_error("Too many tensors");
    }
}

Tensor callFunction(const std::string &name,
                    const std::vector<Tensor> &tensors) {
    if (name == "sum") {
        return sum(tensors);
    } else if (name == "subtraction") {
        return subtraction(tensors);
    } else if (name == "multiplication") {
        return multiplication(tensors);
    } else if (name == "matrix_transpose") {
        return matrix_transpose(tensors);
    } else if (name == "matrix_multiplication") {
        return matrix_multiplication(tensors);
    } else {
        throw std::runtime_error("Wrong operation name");
    }
}

} // namespace NSTTF
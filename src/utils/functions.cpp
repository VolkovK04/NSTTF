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

Tensor sum(std::vector<Tensor> &tensors) {
    checkNumOfTensors(tensors, 2);

    Tensor arg1 = tensors[0];
    Tensor arg2 = tensors[1];

    checkShape(arg1, arg2);

    Tensor res(arg1.getShape());
    functions::sum.exec(
        gpu::WorkSize(functions::workGroupSize, functions::global_work_size),
        arg1.getGPUBuffer(), arg2.getGPUBuffer(), res.getGPUBuffer(),
        // getSize(arg1.getShape()));
        arg1.getSize());

    return res;
}

Tensor subtraction(std::vector<Tensor> &tensors) {
    checkNumOfTensors(tensors, 2);

    Tensor arg1 = tensors[0];
    Tensor arg2 = tensors[1];

    checkShape(arg1, arg2);

    Tensor res(arg1.getShape());
    functions::subtraction.exec(
        gpu::WorkSize(functions::workGroupSize, functions::global_work_size),
        arg1.getGPUBuffer(), arg2.getGPUBuffer(), res.getGPUBuffer(),
        arg1.getSize());
    return res;
}

Tensor multiplication(std::vector<Tensor> &tensors) {
    checkNumOfTensors(tensors, 2);

    Tensor arg1 = tensors[0];
    Tensor arg2 = tensors[1];

    checkShape(arg1, arg2);

    Tensor res(arg1.getShape());
    functions::multiplication.exec(
        gpu::WorkSize(functions::workGroupSize, functions::global_work_size),
        arg1.getGPUBuffer(), arg2.getGPUBuffer(), res.getGPUBuffer(),
        arg1.getSize());
    return res;
}

// Пока считаем, что подаются только матрицы
Tensor matrix_multiplication(std::vector<Tensor> &tensors) {
    checkNumOfTensors(tensors, 2);

    Tensor arg1 = tensors[0];
    Tensor arg2 = tensors[1];

    std::vector<size_t> arg1Shape = arg1.getShape();
    std::vector<size_t> arg2Shape = arg2.getShape();

    if (arg1Shape.size() != arg2Shape.size()) {
        throw std::runtime_error("Different size");
    }

    size_t shapeSize = arg1Shape.size();

    for (size_t i = 0; i <= shapeSize - 3; i++) {
        if (arg1Shape[i] != arg2Shape[i]) {
            throw std::runtime_error("Different num of tensors");
        }
    }

    if (arg1Shape[shapeSize - 1] != arg2Shape[shapeSize - 2]) {
        throw std::runtime_error("Wrong matrix shape");
    }

    arg1Shape.pop_back();
    arg1Shape.push_back(arg2Shape[shapeSize - 1]);

    Tensor res(arg1Shape);
    functions::matrix_multiplication.exec(
        gpu::WorkSize(functions::workGroupSize, functions::global_work_size),
        arg1.getGPUBuffer(), arg2.getGPUBuffer(), res.getGPUBuffer(),
        arg1Shape[shapeSize - 2], arg2Shape[shapeSize - 2],
        arg2Shape[shapeSize - 1]);
    return res;
}

// Пока считаем, что подаются только матрицы
Tensor matrix_transpose(std::vector<Tensor> &tensors) {
    checkNumOfTensors(tensors, 2);

    Tensor arg = tensors[0];

    std::vector<size_t> baseShape = arg.getShape();
    size_t shapeSize = baseShape.size();

    if (shapeSize < 2) {
        throw std::runtime_error("Wrong shape");
    }

    size_t rowCount = baseShape[shapeSize - 2],
           columnCount = baseShape[shapeSize - 1];

    baseShape.pop_back();
    baseShape.pop_back();
    baseShape.push_back(columnCount);
    baseShape.push_back(rowCount);

    Tensor res(baseShape);
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

void checkNumOfTensors(const std::vector<Tensor> &tensors, size_t num) {
    if (tensors.size() < num) {
        throw std::runtime_error("Not enought tensors");
    } else if (tensors.size() > num) {
        throw std::runtime_error("Too many tensors");
    }
}

Tensor callFunction(std::string &name, std::vector<Tensor> &tensors) {
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
    }
}

} // namespace NSTTF
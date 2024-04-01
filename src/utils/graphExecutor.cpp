#include "graphExecutor.h"

namespace NSTTF {

std::string Instruction::getName() { return name; }

std::vector<const std::string> Instruction::getInputs() { return input; }

std::vector<const std::string> Instruction::getOutputs() { return output; }

void GraphExecutor::init() {
    funcMap["subtraction"] = subtraction(
        subtraction_kernel, subtraction_kernel_length, "subtraction");
    funcMap["sum"] = sum(sum_kernel, sum_kernel_length, "sum");
    funcMap["multiplication"] = multiplication(
        multiplication_kernel, multiplication_kernel_length, "multiplication");
    funcMap["matrix_multiplication_updated"] = matrix_multiplication(
        matrix_multiplication_kernel, matrix_multiplication_kernel_length,
        "matrix_multiplication_updated");
    funcMap["matrix_transpose"] =
        matrix_transpose(matrix_transpose_kernel,
                         matrix_transpose_kernel_length, "matrix_transpose");
    subtraction.compile();
    sum.compile();
    multiplication.compile();
    matrix_multiplication.compile();
    matrix_transpose.compile();
}

void GraphExecutor::execute(std::map<std::string, Tensor> &tensorsMap) {
    std::vector<Tensor> tensors;
    for (auto instruction : instructions) {
        for (std::string input : instruction.getInputs()) {
            // if (tensorsMap.find(input) != tensorsMap.end()) {
            //     throw ;
            // }
            tensors.push_back(tensorsMap[input]);
        }
        funcMap[instruction.getName()](tensors);
    }
}
} // namespace NSTTF
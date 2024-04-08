#include "graphExecutor.h"

namespace NSTTF {

std::string Instruction::getName() { return name; }

std::vector<std::string> Instruction::getInputs() { return input; }

std::vector<std::string> Instruction::getOutputs() { return output; }

void GraphExecutor::execute(std::map<std::string, Tensor> &tensorsMap) {
    std::vector<Tensor> tensors;
    for (auto instruction : instructions) {
        std::vector<std::string> inputs = instruction.getInputs();
        for (auto input : inputs) {
            tensors.push_back(tensorsMap[input]);
        }
    }
    //TODO call cl func
}
} // namespace NSTTF
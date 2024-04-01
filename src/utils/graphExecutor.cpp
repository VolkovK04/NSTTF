#include "graphExecutor.h"

namespace NSTTF {

std::string Instruction::getName() { return name; }

std::vector<const std::string> Instruction::getInputs() { return input; }

std::vector<const std::string> Instruction::getOutputs() { return output; }

void GraphExecutor::execute(std::map<std::string, Tensor> &tensorsMap) {
    std::vector<Tensor> tensors;
    for (auto instruction : instructions) {
        for (std::string input : instruction.getInputs()) {
            // if (tensorsMap.find(input) != tensorsMap.end()) {
            //     throw ;
            // }
            tensors.push_back(tensorsMap[input]);
        }
    }
}
} // namespace NSTTF
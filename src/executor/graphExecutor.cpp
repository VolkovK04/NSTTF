#include "graphExecutor.h"

namespace NSTTF {

std::string Instruction::getName() { return name; }

std::vector<std::string> Instruction::getInputs() { return input; }

std::vector<std::string> Instruction::getOutputs() { return output; }

GraphExecutor::GraphExecutor(std::vector<Instruction> instructions)
    : instructions(instructions) {}

std::map<std::string, Tensor>
GraphExecutor::execute(const std::map<std::string, Tensor> &tensorsMap) {
  std::map<std::string, Tensor> updatedMap = tensorsMap;
  std::vector<Tensor> tensors;
  for (auto instruction : instructions) {
    std::vector<std::string> inputs = instruction.getInputs();
    for (auto input : inputs) {
      tensors.push_back(updatedMap[input]);
    }
    updatedMap[instruction.getOutputs()[0]] =
        callFunction(instruction.getName(), tensors);
  }
  return updatedMap;
}
} // namespace NSTTF
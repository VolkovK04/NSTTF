#include "graphExecutor.h"

namespace NSTTF {

std::string Instruction::getName() { return name; }

std::vector<std::string> Instruction::getInputs() { return input; }

std::vector<std::string> Instruction::getOutputs() { return output; }

GraphExecutor::GraphExecutor(std::vector<Instruction> instructions,
                             std::vector<AbstractNode *> outputs)
    : instructions(instructions), outputs(outputs) {}

TensorMap GraphExecutor::execute(const TensorMap &tensorsMap) {
  TensorMap updatedMap = tensorsMap;
  TensorMap outputMap;
  std::vector<Tensor> tensors;
  for (Instruction instruction : instructions) {
    std::vector<std::string> inputs = instruction.getInputs();
    for (std::string input : inputs) {
      tensors.push_back(updatedMap[input]);
    }
    std::vector<std::string> outputNames = instruction.getOutputs();
    for (size_t i = 0; i < outputs.size(); i++) {
      updatedMap[outputNames[i]] = callFunction(instruction.getName(), tensors);
    }
  }
  for (AbstractNode *output : outputs) {
    std::string name = output->getName();
    outputMap[name] = updatedMap[name];
  }
  return outputMap;
}
} // namespace NSTTF
#include "graphExecutor.h"
#include <operations/function.h>

namespace NSTTF {

std::string Instruction::getName() { return name; }

std::vector<std::string> Instruction::getInputs() { return input; }

std::vector<std::string> Instruction::getOutputs() { return output; }

GraphExecutor::GraphExecutor(const std::vector<Instruction> &instructions,
                             const std::vector<AbstractNode *> &outputs)
    : instructions(instructions), outputs(outputs) {}

GraphExecutorWG::GraphExecutorWG(const std::vector<Instruction> &instructions,
                                 const std::vector<AbstractNode *> &outputs,
                                 const std::vector<Instruction> &gradient)
    : GraphExecutor::GraphExecutor(instructions, outputs), gradient(gradient) {}

TensorMap GraphExecutor::execute(const TensorMap &tensorsMap) {
  calculated = tensorsMap;

  for (Instruction instruction : instructions) {
    std::vector<Tensor> tensors;

    std::vector<std::string> inputs = instruction.getInputs();
    for (std::string input : inputs) {
      tensors.push_back(calculated[input]);
    }

    std::vector<Tensor> result = functions.at(instruction.getName())->compute(tensors);

    std::vector<std::string> outputNames = instruction.getOutputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
      calculated[outputNames[i]] = result[i];
    }
  }
  TensorMap outputMap;
  for (AbstractNode *output : outputs) {
    std::string name = output->getName();
    outputMap[name] = calculated[name];
  }
  return std::move(outputMap);
}
} // namespace NSTTF
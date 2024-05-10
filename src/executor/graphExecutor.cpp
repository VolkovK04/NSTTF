#include "graphExecutor.h"
#include <operations/function.h>

namespace NSTTF {

GraphExecutor::GraphExecutor(
    const std::vector<AbstractInstruction *> &instructions,
    const std::vector<AbstractNode *> &outputs)
    : instructions(instructions), outputs(outputs) {}

GraphExecutor::~GraphExecutor() {
  for (auto instruction : instructions) {
    delete instruction;
  }
}

GraphExecutorWG::GraphExecutorWG(
    const std::vector<AbstractInstruction *> &instructions,
    const std::vector<AbstractNode *> &outputs,
    const std::vector<AbstractInstruction *> &gradient)
    : GraphExecutor::GraphExecutor(instructions, outputs), gradient(gradient) {}

TensorMap GraphExecutor::execute(const TensorMap &tensorsMap) {
  calculated = tensorsMap;

  for (AbstractInstruction *abstractInstruction : instructions) {
    Instruction *instruction = dynamic_cast<Instruction *>(abstractInstruction);
    std::string instructionName = instruction->getName();
    if (instructionName == "copy") {
      std::string outputName = instruction->getOutput();
      calculated[outputName] = calculated[instruction->getInputs()[0]];
      continue;
    }

    std::vector<Tensor> tensors;

    std::vector<std::string> inputs = instruction->getInputs();
    for (std::string input : inputs) {
      tensors.push_back(calculated[input]);
    }
    Tensor result = functions.at(instructionName)->compute(tensors);
    std::string outputName = instruction->getOutput();
    calculated[outputName] = result;
  }
  TensorMap outputMap;
  for (AbstractNode *output : outputs) {
    std::string name = output->getName();
    outputMap[name] = calculated[name];
  }
  return std::move(outputMap);
}
} // namespace NSTTF
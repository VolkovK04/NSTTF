#include "compiler.h"
#include "../operations/function.h"

namespace NSTTF {

Instruction::Instruction(const std::string &name,
                         const std::vector<std::string> &input,
                         const std::vector<std::string> &output)
    : name(name), input(input), output(output) {}

Constant::Constant(const std::vector<std::string> &input,
                   const std::vector<std::string> &output, double value)
    : Instruction("constant", input, output), value(value) {}

GraphExecutor Compiler::compile(const ComputationGraph &graph) {
  outputs = graph.getOutputNodes();
  std::vector<Instruction> instructions = getAllInstructions();

  GraphExecutor executor(instructions, outputs);

  return executor;
}

Instruction getDerivative(AbstractNode *node,
                          const std::string differentiateBy) {
  std::string name = node->getName();
  std::vector<std::string> prevNodes;
  std::vector<std::string> nextNodes;
  for (auto prev : node->getPreviousNodes()) {
    prevNodes.push_back(prev->getName());
  }

  for (auto next : node->getNextNodes()) {
    nextNodes.push_back(next->getName());
  }

  if (differentiateBy == name) {
    return Constant(std::move(prevNodes), {name}, 1);
  }
  OperationNode *opNode = dynamic_cast<OperationNode *>(node);
  if (!opNode) {
    return Constant(std::move(prevNodes), {name}, 0);
  }
  std::string opName = opNode->getOperation().getName();
  // functions.at(opName)->derivative();
  if (opName == "sum") {
    for (auto prevNode : prevNodes) {
      if (prevNode == differentiateBy) {
        return Constant(std::move(prevNodes), {name}, 1);
      }
      return Constant(std::move(prevNodes), {name}, 0);
    }
  } else if (opName == "multiplication") {
    if (prevNodes[0] == differentiateBy) {
      return Instruction();
    }
  }
}

void Compiler::getInstruction(AbstractNode *node,
                              std::vector<Instruction> &result) {
  if (computed.count(node)) {
    return;
  }

  OperationNode *operationNode = dynamic_cast<OperationNode *>(node);
  if (!operationNode) {
    return;
  }

  std::vector<std::string> prevNodes;
  std::vector<std::string> nextNodes;

  for (auto prev : operationNode->getPreviousNodes()) {
    prevNodes.push_back(prev->getName());
  }

  for (auto next : operationNode->getNextNodes()) {
    nextNodes.push_back(next->getName());
  }
  Instruction instruction(operationNode->getOperation().getName(),
                          std::move(prevNodes), {node->getName()});
  result.push_back(instruction);
  computed.insert(operationNode);
}

std::vector<Instruction> Compiler::getAllInstructions() {
  std::vector<Instruction> result;

  for (AbstractNode *out : outputs) {
    getInstruction(out, result);
  }

  return result;
}

// GraphExecutorWG Compiler::compile(const ComputationGraph &graph,
//                                   const std::vector<std::string> &inputs) {
//   if (inputs.empty()) {
//     return;
//   }
//   std::unordered_set<std::string> computedInputs;
// }

} // namespace NSTTF
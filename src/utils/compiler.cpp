#include "compiler.h"

namespace NSTTF {

Instruction::Instruction(const std::string &name,
                         const std::vector<std::string> &input,
                         const std::vector<std::string> &output)
    : name(name), input(input), output(output) {}

GraphExecutor Compiler::compile(const ComputationGraph &graph) {
  outputs = graph.getOutputNodes();
  std::vector<Instruction> instructions = getAllInstructions();

  GraphExecutor executor(instructions, outputs);

  return executor;
}

// Instruction getDerivative(AbstractNode *node, const std::vector<std::string>
// differentiateBy) {
//   std::string name = node->getName();
//   for (std::string diffBy : differentiateBy) {
//     if (diffBy == name) {
//       return;
//     }
//   }
//   if (!dynamic_cast<OperationNode *>(node)){

//   }

// }

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
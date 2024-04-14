#include "compiler.h"
#include "../computationGraph/node.h"
#include "instruction.h"
#include <unordered_set>
#include <vector>

namespace NSTTF {

GraphExecutor Compiler::compile(const ComputationGraph &graph) {
  outputs = graph.getOutputNodes();
  std::vector<Instruction> instructions = getAllInstructions();

  GraphExecutor executor(instructions);

  return executor;
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

} // namespace NSTTF
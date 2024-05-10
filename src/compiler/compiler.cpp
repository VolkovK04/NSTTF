#include "compiler.h"
#include "../operations/function.h"
#include <algorithm>

namespace NSTTF {

Instruction::Instruction(const std::string &name,
                         const std::vector<std::string> &inputNodeNames,
                         const std::string &outputNodeName)
    : AbstractInstruction(name), inputNodeNames(inputNodeNames),
      outputNodeName(outputNodeName) {}

GraphExecutor Compiler::compile(const ComputationGraph &graph) {
  outputs = graph.getOutputNodes();
  std::vector<AbstractInstruction *> instructions = getAllInstructions();

  GraphExecutor executor(instructions, outputs);

  return executor;
}

void Compiler::getInstruction(AbstractNode *node,
                              std::vector<AbstractInstruction *> &result) {
  if (computed.count(node)) {
    return;
  }

  OperationNode *operationNode = dynamic_cast<OperationNode *>(node);
  if (!operationNode) {
    return;
  }

  std::vector<std::string> prevNodes;

  for (auto prev : operationNode->getPreviousNodes()) {
    prevNodes.push_back(prev->getName());
  }

  Instruction *instruction = new Instruction(
      operationNode->getOperation(), std::move(prevNodes), node->getName());
  result.push_back(instruction);
  computed.insert(operationNode);
}

std::vector<AbstractInstruction *> Compiler::getAllInstructions() {
  std::vector<AbstractInstruction *> result;

  for (AbstractNode *out : outputs) {
    getInstruction(out, result);
  }

  return result;
}

void getGrads(AbstractNode *node, std::vector<AbstractInstruction *> &result) {
  // it's a HUGE piece of shit xDDDDDDD
  std::vector<AbstractNode *> nextNodes = node->getNextNodes();
  std::string resultName = "~grad_" + node->getName();

  for (auto nextNode : nextNodes) {
    getGrads(nextNode, result);
  }

  for (size_t i = 0; i < nextNodes.size(); ++i) {
    AbstractNode *nextNode = nextNodes[i];
    std::vector<AbstractNode *> nextPrevNodes = nextNode->getPreviousNodes();
    size_t index = std::find(nextPrevNodes.begin(), nextPrevNodes.end(), node) -
                   nextPrevNodes.begin();
    OperationNode *opNode = dynamic_cast<OperationNode *>(node);
    if (!opNode) {
      throw std::runtime_error("Unlucky ;(");
    }
    std::vector<std::string> prevNodes;

    for (auto prev : opNode->getPreviousNodes()) {
      prevNodes.push_back(prev->getName());
    }

    std::string gradName = "~grad_" + nextNode->getName();
    std::vector<AbstractInstruction *> inst =
        functions.at(opNode->getOperation())
            ->derivative(prevNodes, index, gradName, "tmp");
    result.insert(result.end(), inst.begin(), inst.end());
    if (i == 0) {
      result.push_back(new Instruction("copy", {"tmp"}, resultName));
    } else {
      result.push_back(new Instruction("sum", {"tmp", resultName}, resultName));
    }
  }
}

std::vector<AbstractInstruction *>
getAllGrads(const ComputationGraph &graph,
            const std::vector<std::string> &inputs) {
  std::unordered_map<std::string, AbstractNode *> nodeMap = graph.getNodeMap();
  std::vector<AbstractInstruction *> instructions;

  instructions.push_back(new ConstInstruction(
      "const", Tensor(std::vector<float>{1.f}), "~grad_loss"));
  for (std::string input : inputs) {
    AbstractNode *node = nodeMap.at(input);
    getGrads(node, instructions);
  }

  for (InputNode *inputNode : graph.getInputNodes()) {
    inputNode->getName();
  }
}

GraphExecutorWG
Compiler::compileWithGrads(const ComputationGraph &graph,
                           const std::vector<std::string> &inputs) {
  std::unordered_set<std::string> computedInputs;

  std::vector<AbstractInstruction *> instructions = getAllInstructions();
}

} // namespace NSTTF
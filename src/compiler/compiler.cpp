#include "compiler.h"
#include "../operations/function.h"
#include <algorithm>

namespace NSTTF {

void getInstruction(AbstractNode *node,
                    std::vector<AbstractInstruction *> &result,
                    std::unordered_set<AbstractNode *> &computed) {
  if (computed.count(node)) {
    return;
  }

  OperationNode *operationNode = dynamic_cast<OperationNode *>(node);
  if (!operationNode) {
    computed.insert(node);
    return;
  }

  std::vector<std::string> prevNodes;

  for (auto prev : operationNode->getPreviousNodes()) {
    prevNodes.push_back(prev->getName());
    getInstruction(prev, result, computed);
  }

  Instruction *instruction = new Instruction(
      operationNode->getOperation(), std::move(prevNodes), node->getName());
  result.push_back(instruction);
  computed.insert(operationNode);
}

std::vector<AbstractInstruction *>
getAllInstructions(std::vector<AbstractNode *> outputs) {
  std::vector<AbstractInstruction *> result;
  std::unordered_set<AbstractNode *> computed;

  for (AbstractNode *out : outputs) {
    getInstruction(out, result, computed);
  }

  return result;
}

void computeGrads(AbstractNode *node,
                  std::vector<AbstractInstruction *> &result,
                  std::unordered_set<AbstractNode *> &computed) {
  // it's a HUGE piece of shit xDDDDDDD
  if (computed.count(node)) {
    return;
  }

  std::vector<AbstractNode *> nextNodes = node->getNextNodes();
  std::string resultName = "~grad_" + node->getName();

  for (auto nextNode : nextNodes) {
    computeGrads(nextNode, result, computed);
  }

  bool connectedToLoss = false;
  bool first = true;

  for (AbstractNode *nextNode : nextNodes) {

    if (!computed.count(nextNode)) {
      continue;
    }
    connectedToLoss = true;

    std::vector<AbstractNode *> nextPrevNodes = nextNode->getPreviousNodes();
    size_t index = std::find(nextPrevNodes.begin(), nextPrevNodes.end(), node) -
                   nextPrevNodes.begin();

    std::vector<std::string> prevNodes;

    for (auto prev : nextNode->getPreviousNodes()) {
      prevNodes.push_back(prev->getName());
    }
    std::string gradName = "~grad_" + nextNode->getName();

    std::vector<AbstractInstruction *> inst;

    InputNode *inputNode = dynamic_cast<InputNode *>(nextNode);
    if (inputNode) {
      inst = {new Instruction("copy", {gradName}, "tmp")};
    } else {
      OperationNode *opNode = dynamic_cast<OperationNode *>(nextNode);
      if (!opNode) {
        throw std::runtime_error("Unlucky ;(");
      }

      inst = functions.at(opNode->getOperation())
                 ->derivative(prevNodes, index, gradName, "tmp");
    }
    result.insert(result.end(), inst.begin(), inst.end());
    if (first) {
      result.push_back(new Instruction("copy", {"tmp"}, resultName));
      first = false;
    } else {
      result.push_back(new Instruction("sum", {"tmp", resultName}, resultName));
    }
  }

  if (connectedToLoss) {
    computed.insert(node);
  }
}

std::vector<AbstractInstruction *>
computeAllGrads(const ComputationGraph &graph,
                const std::vector<std::string> &inputs) {
  std::unordered_map<std::string, AbstractNode *> nodeMap = graph.getNodeMap();
  std::vector<AbstractInstruction *> instructions;

  instructions.push_back(new ConstInstruction(
      "const", Tensor(std::vector<float>{1.f}), "~grad_loss"));

  std::unordered_set<AbstractNode *> computed;
  computed.insert(nodeMap.at("loss"));

  for (std::string input : inputs) {
    AbstractNode *node = nodeMap.at(input);
    computeGrads(node, instructions, computed);
  }

  for (InputNode *inputNode : graph.getInputNodes()) {
    inputNode->getName();
  }
  return std::move(instructions);
}

Instruction::Instruction(const std::string &name,
                         const std::vector<std::string> &inputNodeNames,
                         const std::string &outputNodeName)
    : AbstractInstruction(name), inputNodeNames(inputNodeNames),
      outputNodeName(outputNodeName) {}

GraphExecutor Compiler::compile(const ComputationGraph &graph) {
  std::vector<AbstractNode *> outputs = graph.getOutputNodes();
  std::vector<AbstractInstruction *> instructions = getAllInstructions(outputs);

  size_t size = outputs.size();
  std::vector<std::string> outputsNames(size);

  for (size_t i = 0; i < size; ++i) {
    outputsNames[i] = outputs[i]->getName();
  }

  GraphExecutor executor(instructions, outputsNames);

  return executor;
}

GraphExecutorWG
Compiler::compileWithGrads(const ComputationGraph &graph,
                           const std::vector<std::string> &inputs) {

  std::vector<AbstractInstruction *> instructions =
      getAllInstructions(graph.getOutputNodes());

  std::vector<AbstractInstruction *> gradInstructions =
      computeAllGrads(graph, inputs);

  std::vector<AbstractNode *> outputs = graph.getOutputNodes();
  size_t size = outputs.size();
  std::vector<std::string> outputNames(size);

  for (size_t i = 0; i < size; ++i) {
    outputNames[i] = outputs[i]->getName();
  }

  return GraphExecutorWG(instructions, outputNames, inputs, gradInstructions);
}

GraphExecutorWG Compiler::compileWithGrads(const ComputationGraph &graph) {
  std::vector<std::string> inputs;
  for (InputNode *inputNode : graph.getInputNodes()) {
    inputs.push_back(inputNode->getName());
  }
  return Compiler::compileWithGrads(graph, inputs);
}

} // namespace NSTTF
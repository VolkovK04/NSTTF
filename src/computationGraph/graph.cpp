#include "computationGraph.h"
#include "node.h"
#include <unordered_set>

namespace NSTTF {
InputNode &ComputationGraph::AddInputNode() {
  InputNode *node = new InputNode();
  this->input.push_back(node);
  return *node;
}

OperationNode &
ComputationGraph::AddOperationNode(const AbstractOperation operation,
                                   const std::vector<AbstractNode *> &nodes,
                                   bool output = false) {
  OperationNode *node = new OperationNode();
  node->prevs = nodes;
  node->operation = operation;
  node->output = output;
  for (auto node : nodes) {
    node->nexts.push_back(node);
  }
  if (output) {
    this->output.push_back(node);
  }
  return *node;
}

const std::vector<InputNode *> ComputationGraph::getInputNodes() const {
  return input;
}

const std::vector<AbstractNode *> ComputationGraph::getOutputNodes() const {
  return output;
}

void ComputationGraph::getAllNextNodes(
    AbstractNode *node, std::unordered_set<AbstractNode *> &output) const {
  if (output.find(node) == output.end()) {
    return;
  }
  output.insert(node);
  for (auto next : node->nexts) {
    getAllNextNodes(next, output);
  }
}

std::unordered_set<AbstractNode *> ComputationGraph::getAllNodes() const {
  std::unordered_set<AbstractNode *> set;
  for (InputNode *node : input) {
    getAllNextNodes(node, set);
  }
  return set;
}

ComputationGraph::~ComputationGraph() {
  std::unordered_set<AbstractNode *> set = getAllNodes();
  for (auto node : set) {
    delete node;
  }
}
} // namespace NSTTF
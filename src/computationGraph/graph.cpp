#include "computationGraph.h"
#include <unordered_set>

namespace NSTTF {
NodeInterface ComputationGraph::AddInputNode(const std::string &name) {
  InputNode *node = new InputNode();
  node->name = name;
  this->input.push_back(node);
  nodeMap.insert({name, node});
  return NodeInterface(node, *this);
}

NodeInterface ComputationGraph::AddOperationNode(
    const std::string &operationName, const std::vector<AbstractNode *> &nodes,
    const std::string &name, bool output = false) {
  OperationNode *node = new OperationNode();
  node->prevs = nodes;
  node->operationName = operationName;
  node->name = name;

  for (AbstractNode *prev : nodes) {
    prev->nexts.push_back(node);
  }
  if (output) {
    setOutputNode(node);
  }
  nodeMap.insert({name, node});
  return NodeInterface(node, *this);
}

void ComputationGraph::renameNode(const std::string &oldName,
                                  const std::string &newName) {
  auto node = nodeMap.at(oldName);
  nodeMap.erase(oldName);
  node->name = newName;
  nodeMap.insert({newName, node});
}

void ComputationGraph::setOutputNode(AbstractNode *node) {
  node->output = true;
  output.push_back(node);
}

const std::vector<InputNode *> ComputationGraph::getInputNodes() const {
  return input;
}

const std::vector<AbstractNode *> ComputationGraph::getOutputNodes() const {
  return output;
}

void ComputationGraph::getAllNextNodes(
    AbstractNode *node, std::unordered_set<AbstractNode *> &set) const {
  if (set.find(node) != set.end()) {
    return;
  }
  set.insert(node);
  for (AbstractNode *next : node->nexts) {
    getAllNextNodes(next, set);
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
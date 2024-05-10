#pragma once

#include "node.h"
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace NSTTF {
class ComputationGraph {
  friend NodeInterface;

private:
  std::vector<InputNode *> input;
  std::vector<AbstractNode *> output;
  std::unordered_map<std::string, AbstractNode *> nodeMap;

  void getAllNextNodes(AbstractNode *node,
                       std::unordered_set<AbstractNode *> &output) const;

  std::unordered_set<AbstractNode *> getAllNodes() const;

  void setOutputNode(AbstractNode *node);

public:
  ComputationGraph() = default;
  ~ComputationGraph();

  const std::vector<InputNode *> getInputNodes() const;

  const std::vector<AbstractNode *> getOutputNodes() const;

  NodeInterface AddInputNode(const std::string &name);
  NodeInterface AddOperationNode(const std::string &operationName,
                                 const std::vector<AbstractNode *> &nodes,
                                 const std::string &name, bool output);

  const std::unordered_map<std::string, AbstractNode *> &getNodeMap() const {
    return nodeMap;
  }
};
} // namespace NSTTF
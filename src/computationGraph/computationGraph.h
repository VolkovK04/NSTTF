#pragma once
#include "node.h"
#include <iostream>
#include <unordered_set>
#include <vector>


namespace NSTTF {
class ComputationGraph {
private:
  std::vector<InputNode *> input;
  std::vector<AbstractNode *> output;

  void getAllNextNodes(AbstractNode *node,
                       std::unordered_set<AbstractNode *> &output) const;

  const std::vector<InputNode *> getInputNodes() const;

  std::unordered_set<AbstractNode *> ComputationGraph::getAllNodes() const;

public:
  ComputationGraph() = default;
  ~ComputationGraph();

  InputNode &AddInputNode();
  OperationNode &AddOperationNode(const AbstractOperation operation,
                                  const std::vector<AbstractNode *> &nodes,
                                  bool output);
};
} // namespace NSTTF
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

    std::unordered_set<AbstractNode *> getAllNodes() const;

  public:
    ComputationGraph() = default;
    ~ComputationGraph();

    const std::vector<InputNode *> getInputNodes() const;

    const std::vector<AbstractNode *> getOutputNodes() const;

    InputNode &AddInputNode(const std::string& name);
    OperationNode &AddOperationNode(const AbstractOperation& operation,
                                    const std::vector<AbstractNode *> &nodes,
                                    const std::string& name,
                                    bool output);
};
} // namespace NSTTF
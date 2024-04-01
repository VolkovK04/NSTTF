#pragma once

#include "../operations/abstractOperation.h"
#include "computationGraph.h"
#include <vector>

namespace NSTTF {

class ComputationGraph;

class AbstractNode {
    friend class ComputationGraph;

  protected:
    std::vector<AbstractNode *> prevs;
    std::vector<AbstractNode *> nexts;
    std::string name;
    bool output = false;

  public:
    std::vector<AbstractNode *> getPreviousNodes();
    std::vector<AbstractNode *> getNextNodes();
    std::string getName();

    AbstractNode() = default;

    virtual ~AbstractNode() = default;
};

class OperationNode : public AbstractNode {
    friend class ComputationGraph;

  private:
    AbstractOperation operation;

  public:
    OperationNode() = default;
    OperationNode(const AbstractOperation operation,
                  const std::vector<AbstractNode *> &nodes);
    ~OperationNode() = default;

    AbstractOperation getOperation();
};

class InputNode : public AbstractNode {

  public:
    InputNode() = default;
    ~InputNode() = default;
};

} // namespace NSTTF

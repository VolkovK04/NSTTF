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
  bool output = false;

public:
  AbstractNode() = default;

  std::vector<AbstractNode *> getPreviousNodes() 
  {
    return prevs;
  }

  std::vector<AbstractNode *> getNextNodes() 
  {
    return nexts;
  }

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
};

class InputNode : public AbstractNode {

public:
  InputNode() = default;
  ~InputNode() = default;
};

} // namespace NSTTF

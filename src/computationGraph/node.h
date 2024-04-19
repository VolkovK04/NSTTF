#pragma once

#include <cstdint>

#include <operations/abstractOperation.h>
#include <vector>

namespace NSTTF {

class ComputationGraph;

class AbstractNode {
  friend class ComputationGraph;
  friend class NodeInterface;

protected:
  std::vector<AbstractNode *> prevs;
  std::vector<AbstractNode *> nexts;
  std::string name;
  bool output = false;

public:
  std::vector<AbstractNode *> getPreviousNodes() const;
  std::vector<AbstractNode *> getNextNodes() const;
  const std::string &getName() const;

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

  AbstractOperation getOperation() const;
};

class InputNode : public AbstractNode {

public:
  InputNode() = default;
  ~InputNode() = default;
};

class NodeInterface {
private:
  AbstractNode *node;
  ComputationGraph &graph;

  static uint64_t UID_Counter;
  static std::string createName();
  static void checkSameGraph(NodeInterface i1, NodeInterface i2);

public:
  NodeInterface(AbstractNode *node, ComputationGraph &g);

  const AbstractNode &getNode() const;
  void setOutput();
  void setName(const std::string &name);

  NodeInterface operator+(const NodeInterface &nodeInterface) const;
  NodeInterface operator-(const NodeInterface &nodeInterface) const;
  NodeInterface operator*(const NodeInterface &nodeInterface) const;

  static NodeInterface MatrixMult(const NodeInterface &left,
                                  const NodeInterface &right);
  static NodeInterface MatrixTranspose(const NodeInterface &node);
};

} // namespace NSTTF

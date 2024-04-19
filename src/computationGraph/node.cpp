#include "node.h"

#include "computationGraph.h"

namespace NSTTF {

uint64_t NodeInterface::UID_Counter = 0;

std::vector<AbstractNode *> AbstractNode::getPreviousNodes() const {
  return prevs;
}

std::vector<AbstractNode *> AbstractNode::getNextNodes() const { return nexts; }

const std::string &AbstractNode::getName() const { return name; }

AbstractOperation OperationNode::getOperation() const { return operation; }

const AbstractNode &NodeInterface::getNode() const { return *node; }

void NodeInterface::setOutput() { graph.setOutputNode(node); }

void NodeInterface::setName(const std::string &name) { node->name = name; }

NodeInterface::NodeInterface(AbstractNode *node, ComputationGraph &g)
    : graph(g), node(node) {}

std::string NodeInterface::createName() {
  return "~" + std::to_string(UID_Counter);
}

NodeInterface
NodeInterface::operator+(const NodeInterface &nodeInterface) const {
  checkSameGraph(*this, nodeInterface);
  AbstractOperation sum("sum");
  std::string name = createName();
  return graph.AddOperationNode(sum, {node, nodeInterface.node}, name, false);
}

NodeInterface
NodeInterface::operator-(const NodeInterface &nodeInterface) const {
  checkSameGraph(*this, nodeInterface);
  AbstractOperation subtraction("subtraction");
  std::string name = createName();
  return graph.AddOperationNode(subtraction, {node, nodeInterface.node}, name,
                                false);
}

NodeInterface
NodeInterface::operator*(const NodeInterface &nodeInterface) const {
  checkSameGraph(*this, nodeInterface);
  AbstractOperation multiplication("multiplication");
  std::string name = createName();
  return graph.AddOperationNode(multiplication, {node, nodeInterface.node},
                                name, false);
}

NodeInterface NodeInterface::MatrixMult(const NodeInterface &left,
                                        const NodeInterface &right) {
  checkSameGraph(left, right);
  AbstractOperation multiplication("matrix_multiplication");
  std::string name = createName();
  return left.graph.AddOperationNode(multiplication, {left.node, right.node},
                                     name, false);
}

NodeInterface NodeInterface::MatrixTranspose(const NodeInterface &node) {
  AbstractOperation multiplication("matrix_transpose");
  std::string name = createName();
  return node.graph.AddOperationNode(multiplication, {node.node}, name, false);
}

void NodeInterface::checkSameGraph(NodeInterface i1, NodeInterface i2) {
  if (&i1.graph != &i2.graph) {
    throw std::runtime_error("Nodes defined in different graphs");
  }
}

} // namespace NSTTF
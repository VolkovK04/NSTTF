#include "node.h"

#include "computationGraph.h"

namespace NSTTF {

uint64_t NodeInterface::UID_Counter = 0;

std::vector<AbstractNode *> AbstractNode::getPreviousNodes() { return prevs; }

std::vector<AbstractNode *> AbstractNode::getNextNodes() { return nexts; }

const std::string& AbstractNode::getName() const { return name; }

AbstractOperation OperationNode::getOperation() const { return operation; }

const AbstractNode& NodeInterface::getNode() const {
    return *node;
}

void NodeInterface::setOutput() {
    graph.setOutputNode(node);
}

NodeInterface::NodeInterface(AbstractNode* node, ComputationGraph& g): graph(g), node(node) {}

std::string NodeInterface::createName() {
    return "~" + std::to_string(UID_Counter);
}

NodeInterface NodeInterface::operator+ (const NodeInterface& nodeInterface) const
{
    // TODO check if nodes in different graphs
    AbstractOperation sum("sum");
    std::string name = createName();
    return graph.AddOperationNode(sum, {node, nodeInterface.node}, name, false);
}

NodeInterface NodeInterface::operator- (const NodeInterface& nodeInterface) const
{
    // TODO check if nodes in different graphs
    AbstractOperation subtraction("subtraction");
    std::string name = createName();
    return graph.AddOperationNode(subtraction, {node, nodeInterface.node}, name, false);
}

NodeInterface NodeInterface::operator* (const NodeInterface& nodeInterface) const
{
    // TODO check if nodes in different graphs
    AbstractOperation multiplication("multiplication");
    std::string name = createName();
    return graph.AddOperationNode(multiplication, {node, nodeInterface.node}, name, false);
}



} // namespace NSTTF
#include "node.h"

namespace NSTTF {
std::vector<AbstractNode *> AbstractNode::getPreviousNodes() { return prevs; }

std::vector<AbstractNode *> AbstractNode::getNextNodes() { return nexts; }

std::string AbstractNode::getName() { return name; }

AbstractOperation OperationNode::getOperation() { return operation; }

} // namespace NSTTF
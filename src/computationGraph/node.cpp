#include "node.h"

namespace NSTTF {
std::vector<AbstractNode *> AbstractNode::getPreviousNodes() { return prevs; }

std::vector<AbstractNode *> AbstractNode::getNextNodes() { return nexts; }

} // namespace NSTTF
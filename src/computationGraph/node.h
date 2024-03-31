#pragma once
#include "../operations/abstractOperation.h"
#include <vector>

namespace NSTTF {
class AbstractNode {
protected:
  std::vector<AbstractNode *> prevs;
  std::vector<AbstractNode *> nexts;

public:
  AbstractNode() = default;

  virtual ~AbstractNode() = default;
};

class Node : public AbstractNode {
private:
  AbstractOperation *operation;

public:
  Node() = default;
  ~Node() = default;
};

class InputNode : public AbstractNode {

public:
  InputNode() = default;
  ~InputNode() = default;
};

class OutputNode : public AbstractNode {
public:
  ~OutputNode() = default;
};

} // namespace NSTTF

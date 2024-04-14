#pragma once

#include "../computationGraph/computationGraph.h"
#include "../computationGraph/node.h"
#include "graphExecutor.h"

namespace NSTTF {
class Compiler {
private:
  std::unordered_set<AbstractNode *> computed;
  std::vector<AbstractNode *> outputs;

  void getInstruction(AbstractNode *node, std::vector<Instruction> &result);
  std::vector<Instruction> getAllInstructions();

public:
  Compiler() = default;

  GraphExecutor compile(const ComputationGraph &graph);
};
} // namespace NSTTF
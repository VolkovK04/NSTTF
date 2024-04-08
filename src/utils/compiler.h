#pragma once

#include "../computationGraph/computationGraph.h"
#include "../computationGraph/node.h"
#include "graphExecutor.h"

namespace NSTTF {
class Compiler {
private:
  std::unordered_set<AbstractNode *> computed;
  std::vector<AbstractNode *> outputs;

  void get_instruction(AbstractNode *node, std::vector<Instruction> &result);

public:
  Compiler() = default;
  std::vector<Instruction> get_all_instructions();

  GraphExecutor compile(const ComputationGraph &graph);
};
} // namespace NSTTF
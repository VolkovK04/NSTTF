#pragma once

#include "instruction.h"
#include <computationGraph/computationGraph.h>
#include <computationGraph/node.h>
#include <executor/graphExecutor.h>
#include <unordered_set>
#include <vector>

namespace NSTTF {
class Compiler {
private:
  std::unordered_set<AbstractNode *> computed;
  std::vector<AbstractNode *> outputs;

  void getInstruction(AbstractNode *node,
                      std::vector<AbstractInstruction *> &result);
  std::vector<AbstractInstruction *> getAllInstructions();

public:
  Compiler() = default;

  GraphExecutor compile(const ComputationGraph &graph);

  GraphExecutorWG compileWithGrads(const ComputationGraph &graph,
                                   const std::vector<std::string> &inputs);
};
} // namespace NSTTF
#pragma once

#include "instruction.h"
#include <computationGraph/computationGraph.h>
#include <computationGraph/node.h>
#include <executor/graphExecutor.h>
#include <unordered_set>
#include <vector>

namespace NSTTF {
class Compiler {
public:
  Compiler() = default;

  GraphExecutor compile(const ComputationGraph &graph);

  GraphExecutorWG compileWithGrads(const ComputationGraph &graph,
                                   const std::vector<std::string> &inputs);

  GraphExecutorWG compileWithGrads(const ComputationGraph &graph);
};
} // namespace NSTTF
#pragma once

#include <computationGraph/node.h>
#include <map>
#include <tensor/tensor.h>
#include <unordered_set>
#include <utils/functions.h>
#include <utils/instruction.h>
#include <vector>

namespace NSTTF {
class GraphExecutor {
private:
  std::vector<Instruction> instructions;
  std::vector<AbstractNode *> outputs;
  TensorMap gradient;

public:
  GraphExecutor(const std::vector<Instruction> &instructions,
                const std::vector<AbstractNode *> &outputs);
  GraphExecutor(const std::vector<Instruction> &instrucitons,
                const TensorMap &gradient,
                const std::vector<AbstractNode *> &outputs);

  TensorMap execute(const TensorMap &tensorsMap);
};

class GraphExecutorWG : GraphExecutor {};
} // namespace NSTTF
#pragma once

#include <computationGraph/node.h>
#include <map>
#include <tensor/tensor.h>
#include <utils/instruction.h>
#include <vector>

namespace NSTTF {
class GraphExecutor {
private:
  std::vector<Instruction> instructions;
  std::vector<AbstractNode *> outputs;
  TensorMap gradient;

public:
  GraphExecutor(std::vector<Instruction> instructions,
                std::vector<AbstractNode *> outputs);
  GraphExecutor(std::vector<Instruction> instrucitons, TensorMap gradient,
                std::vector<AbstractNode *> outputs);

  TensorMap execute(const TensorMap &tensorsMap);
};
} // namespace NSTTF
#pragma once

#include <computationGraph/node.h>
#include <map>
#include <tensor/tensor.h>
#include <utils/instruction.h>
#include <vector>

namespace NSTTF {
class GraphExecutor {
protected:
  std::vector<Instruction> instructions;
  std::vector<AbstractNode *> outputs;

public:
  GraphExecutor() = default;

  GraphExecutor(const std::vector<Instruction> &instructions,
                const std::vector<AbstractNode *> &outputs);

  TensorMap execute(const TensorMap &tensorsMap);
};

class GraphExecutorWG : public GraphExecutor {
private:
 T
public:
  GraphExecutorWG(const std::vector<Instruction> &Instructions,
                  const ComputationGraph &graph);
};
} // namespace NSTTF
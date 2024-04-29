#pragma once

#include <compiler/instruction.h>
#include <computationGraph/node.h>
#include <map>
#include <tensor/tensor.h>
#include <vector>

namespace NSTTF {
class GraphExecutor {
protected:
  std::vector<Instruction> instructions;
  std::vector<AbstractNode *> outputs;
  TensorMap calculated;

public:
  GraphExecutor() = default;

  GraphExecutor(const std::vector<Instruction> &instructions,
                const std::vector<AbstractNode *> &outputs);

  TensorMap execute(const TensorMap &tensorsMap);
};

class GraphExecutorWG : public GraphExecutor {
private:
  std::vector<Instruction> gradient;

public:
  GraphExecutorWG(const std::vector<Instruction> &instructions,
                  const std::vector<AbstractNode *> &outputs,
                  const std::vector<Instruction> &gradient);
  TensorMap executeGrads(const TensorMap &tensorsMap);
};
} // namespace NSTTF
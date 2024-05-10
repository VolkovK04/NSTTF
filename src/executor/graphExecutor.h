#pragma once

#include <compiler/instruction.h>
#include <computationGraph/node.h>
#include <map>
#include <tensor/tensor.h>
#include <vector>

namespace NSTTF {
class GraphExecutor {
protected:
  std::vector<AbstractInstruction *> instructions;
  std::vector<AbstractNode *> outputs;
  TensorMap calculated;

public:
  GraphExecutor() = default;

  GraphExecutor(const std::vector<AbstractInstruction *> &instructions,
                const std::vector<AbstractNode *> &outputs);

  TensorMap execute(const TensorMap &tensorsMap);

  ~GraphExecutor();
};

class GraphExecutorWG : public GraphExecutor {
private:
  std::vector<AbstractInstruction *> gradient;

public:
  GraphExecutorWG(const std::vector<AbstractInstruction *> &instructions,
                  const std::vector<AbstractNode *> &outputs,
                  const std::vector<AbstractInstruction *> &gradient);
  TensorMap executeGrads(const TensorMap &tensorsMap);
};
} // namespace NSTTF
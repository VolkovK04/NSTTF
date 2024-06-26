#pragma once

#include <compiler/instruction.h>
#include <computationGraph/node.h>
#include <tensor/tensor.h>
#include <vector>

namespace NSTTF {
class GraphExecutor {
protected:
  std::vector<AbstractInstruction *> instructions;
  std::vector<std::string> outputs;
  TensorMap calculated;

public:
  GraphExecutor() = default;

  GraphExecutor(const std::vector<AbstractInstruction *> &instructions,
                const std::vector<std::string> &outputs);

  TensorMap execute(const TensorMap &tensorsMap);

  void printInstructions(std::ostream &stream) const;

  virtual ~GraphExecutor();
};

class GraphExecutorWG : public GraphExecutor {
protected:
  std::vector<AbstractInstruction *> gradInstructions;
  TensorMap grads;
  std::vector<std::string> inputs;

public:
  GraphExecutorWG() = default;
  GraphExecutorWG(const std::vector<AbstractInstruction *> &instructions,
                  const std::vector<std::string> &outputs,
                  const std::vector<std::string> &inputs,
                  const std::vector<AbstractInstruction *> &gradient);
  TensorMap executeGrads();

  void printGradInstructions(std::ostream &stream) const;

  const std::vector<AbstractInstruction *> &getGradInstructions() const;

  ~GraphExecutorWG() override;
};
} // namespace NSTTF
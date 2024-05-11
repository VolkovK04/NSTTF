#pragma once

#include <string>
#include <tensor/tensor.h>
#include <vector>

namespace NSTTF {

class AbstractInstruction {
protected:
  std::string name;

public:
  AbstractInstruction() = default;
  AbstractInstruction(const std::string &name) : name(name) {}
  const std::string &getName() const;
  virtual ~AbstractInstruction() = default;
};

class Instruction : public AbstractInstruction {
protected:
  std::vector<std::string> inputNodeNames;
  std::string outputNodeName;

public:
  const std::vector<std::string> &getInputs() const;
  const std::string &getOutput() const;

  ///
  /// args:
  ///   name - operation that we should execute (THE ONLY ONE!)
  ///   input - input arguments
  ///   output - output argument (THE ONLY ONE!)
  ///
  Instruction(const std::string &name, const std::vector<std::string> &input,
              const std::string &output);

  Instruction() = default;

  ~Instruction() = default;
};

class ConstInstruction : public AbstractInstruction {
protected:
  std::string outputNodeName;
  Tensor tensor;

public:
  ConstInstruction() = default;
  ConstInstruction(const std::string &name, Tensor tensor,
                   const std::string &outputNodeName)
      : AbstractInstruction(name), tensor(tensor),
        outputNodeName(outputNodeName) {}

  const std::string &getOutput() const { return outputNodeName; }
  Tensor getTensor() const { return tensor; }
  ~ConstInstruction() = default;
};

} // namespace NSTTF
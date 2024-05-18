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
  virtual void printInfo(std::ostream &stream) const;
  virtual ~AbstractInstruction() = default;
};

class Instruction : public AbstractInstruction {
protected:
  std::vector<std::string> inputNodeNames;
  std::string outputNodeName;

public:
  const std::vector<std::string> &getInputs() const;
  const std::string &getOutput() const;

  Instruction() = default;

  ///
  /// args:
  ///   name - operation that we should execute (THE ONLY ONE!)
  ///   input - input arguments
  ///   output - output argument (THE ONLY ONE!)
  ///
  Instruction(const std::string &name, const std::vector<std::string> &input,
              const std::string &output);

  void printInfo(std::ostream &stream) const override;

  ~Instruction() = default;
};

class ConstInstruction : public AbstractInstruction {
protected:
  Tensor tensor;
  std::string outputNodeName;

public:
  ConstInstruction() = default;
  ConstInstruction(const std::string &name, Tensor tensor,
                   const std::string &outputNodeName)
      : AbstractInstruction(name), tensor(tensor),
        outputNodeName(outputNodeName) {}

  const std::string &getOutput() const { return outputNodeName; }
  Tensor getTensor() const { return tensor; }
  void printInfo(std::ostream &stream) const override;
  ~ConstInstruction() = default;
};

} // namespace NSTTF
#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <operations/function.h>

namespace NSTTF {
class Instruction {
protected:
  // Function function;
  std::string name;
  std::vector<std::string> input;
  std::vector<std::string> output;

public:
  std::string getName();
  std::vector<std::string> getInputs();
  std::vector<std::string> getOutputs();

  Instruction(const std::string &name, const std::vector<std::string> &input,
              const std::vector<std::string> &output);

  Instruction() = default;

  ~Instruction() = default;
};

class Constant : public Instruction {
private:
  double value;

public:
  Constant(const std::vector<std::string> &input,
           const std::vector<std::string> &output, double value);

  Constant() = default;

  ~Constant() = default;
};

} // namespace NSTTF
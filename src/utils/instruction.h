#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace NSTTF {
class Instruction {
private:
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

} // namespace NSTTF
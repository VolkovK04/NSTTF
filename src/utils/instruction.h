#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace NSTTF {
class Instruction {
  private:
    const std::string name;
    const std::vector<const std::string> input;
    const std::vector<const std::string> output;

  public:
    std::string getName();
    std::vector<const std::string> getInputs();
    std::vector<const std::string> getOutputs();

    Instruction(const std::string &name,
                const std::vector<const std::string> &input,
                const std::vector<const std::string> &output)
        : name(name), output(output), input(input) {}
    ~Instruction() = default;
};
} // namespace NSTTF
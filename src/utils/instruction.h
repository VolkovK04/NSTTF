#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace NSTTF {
class instruction {
  private:
    const std::string name;
    const std::vector<const std::string> input;
    const std::vector<const std::string> output;

  public:
    instruction(const std::string &name,
                const std::vector<const std::string> &input,
                const std::vector<const std::string> &output)
        : name(name), output(output), input(input) {}
    ~instruction() = default;
};
} // namespace NSTTF
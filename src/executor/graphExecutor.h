#pragma once

#include <map>
#include <tensor/tensor.h>
#include <utils/functions.h>
#include <utils/instruction.h>
#include <vector>

namespace NSTTF {
class GraphExecutor {
private:
  std::vector<Instruction> instructions;

public:
  GraphExecutor(std::vector<Instruction> instructions);

  std::map<std::string, Tensor>
  execute(const std::map<std::string, Tensor> &tensorsMap);
};
} // namespace NSTTF
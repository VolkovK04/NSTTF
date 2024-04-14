#pragma once

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "../tensor/tensor.h"
#include "instruction.h"
#include <map>
#include <vector>

namespace NSTTF {
class GraphExecutor {
  private:
    std::vector<Instruction> instructions;

  public:
    GraphExecutor(std::vector<Instruction> instructions);

    std::map<std::string, Tensor> execute(const std::map<std::string, Tensor> &tensorsMap);
};
} // namespace NSTTF
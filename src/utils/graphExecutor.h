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

#include "../cl_build_headers/matrix_multiplication_cl.h"
#include "../cl_build_headers/matrix_transpose_cl.h"
#include "../cl_build_headers/multiplication_cl.h"
#include "../cl_build_headers/subtraction_cl.h"
#include "../cl_build_headers/sum_cl.h"

namespace NSTTF {
class GraphExecutor {
  private:
    std::vector<Instruction> instructions;

  public:
    GraphExecutor(std::vector<Instruction> instructions)
        : instructions(instructions) {}

    

    void init();
    void execute(std::map<std::string, Tensor> &tensorsMap);
};
} // namespace NSTTF
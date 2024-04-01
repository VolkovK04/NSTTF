#pragma once

#include <vector>
#include "instruction.h"
#include <map>
#include "../tensor/tensor.h"
#include <CL/cl.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

namespace NSTTF 
{
    class GraphExecutor 
    {
        private:
        std::map<std::string, ocl::Kernel>
        std::vector<Instruction> instructions;
        void execute(std::map<std::string, Tensor>& tensorsMap);
    };
}
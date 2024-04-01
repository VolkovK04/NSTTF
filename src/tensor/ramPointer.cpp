#include "tensor.h"

namespace NSTTF {

RAMPointer::RAMPointer(std::vector<float>& pointer) : pointer(pointer){};

std::vector<float> RAMPointer::toVector() { return pointer; }

gpu::gpu_mem_32f RAMPointer::toGPUBuffer() {
    gpu::gpu_mem_32f buffer;
    buffer.resizeN(pointer.size());
    buffer.writeN(pointer.data(), pointer.size());
    return buffer;
}

} // namespace NSTTF
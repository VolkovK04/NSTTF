#include "tensor.h"
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

namespace NSTTF {

GPUPointer::GPUPointer(gpu::gpu_mem_32f &pointer) : pointer(pointer) {}
GPUPointer::GPUPointer(const std::vector<float> &vector) {
  pointer.resizeN(vector.size());
  pointer.writeN(vector.data(), vector.size());
}
GPUPointer::GPUPointer(size_t size) { pointer.resizeN(size); }

std::vector<float> GPUPointer::toVector() const {
  std::vector<float> vector;
  pointer.readN(vector.data(), pointer.size());
  return vector;
}

gpu::gpu_mem_32f GPUPointer::toGPUBuffer() const { return pointer; }
} // namespace NSTTF
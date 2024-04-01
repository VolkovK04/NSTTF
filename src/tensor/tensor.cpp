#include "tensor.h"
#include "../utils/utils.h"

namespace NSTTF {
Tensor::Tensor(GPUPointer pointer) : pointer(pointer) {}
gpu::gpu_mem_32f Tensor::getGPUBuffer() { return pointer.toGPUBuffer(); }
Tensor::Tensor(const std::vector<size_t> &shape)
    : shape(shape), pointer(getSize(shape)) {}
std::vector<size_t> Tensor::getShape() {
    return shape;
}
Tensor::Tensor(const std::vector<float>& vector): pointer(vector), shape(vector.size()) {}

} // namespace NSTTF
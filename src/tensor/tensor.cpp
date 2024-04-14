#include "tensor.h"

namespace NSTTF {
Tensor::Tensor(GPUPointer pointer) : pointer(pointer) {}
gpu::gpu_mem_32f Tensor::getGPUBuffer() { return pointer.toGPUBuffer(); }
Tensor::Tensor(const std::vector<size_t> &shape)
    : shape(shape), pointer(getSize(shape)) {}
std::vector<size_t> Tensor::getShape() { return shape; }
Tensor::Tensor(const std::vector<float> &vector)
    : pointer(vector), shape(1, vector.size()) {}

size_t Tensor::getSize() {
    std::vector<size_t> shape = this->getShape();

    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    return size;
}

size_t Tensor::getSize(const std::vector<size_t> &shape) {
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    return size;
}

} // namespace NSTTF
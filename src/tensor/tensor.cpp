#include "tensor.h"

namespace NSTTF {
gpu::gpu_mem_32f Tensor::getGPUBuffer() const { return pointer.toGPUBuffer(); }

Tensor::Tensor(const std::vector<float> &data, const std::vector<size_t> &shape)
    : shape(shape), pointer(data) {
  size_t size = getSize();
  if (size == 0) {
    throw std::invalid_argument("Invalid shape");
  }
  if (data.size() != size) {
    throw std::invalid_argument("Invalid data");
  }
}

Tensor::Tensor(const std::vector<size_t> &shape)
    : shape(shape), pointer(getSize(shape)) {
  if (getSize() == 0) {
    throw std::invalid_argument("Invalid shape");
  }
}
std::vector<size_t> Tensor::getShape() const { return shape; }
Tensor::Tensor(const std::vector<float> &data)
    : Tensor(data, std::vector<size_t>(1, data.size())) {}

size_t Tensor::getSize() const {
  if (shape.size() == 0) {
    return 0;
  }
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

std::vector<float> Tensor::getData() const {
  size_t size = getSize(getShape());
  gpu::gpu_mem_32f buffer = getGPUBuffer();
  std::vector<float> data(size);
  buffer.readN(data.data(), size);
  return data;
}

void Tensor::reshape(const std::vector<size_t> &newShape) {
  if (getSize(newShape) != getSize()) {
    throw std::invalid_argument("Incompatible shapes");
  }
  shape = newShape;
}

} // namespace NSTTF
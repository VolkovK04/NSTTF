#include "tensor.h"

namespace NSTTF {
gpu::gpu_mem_32f Tensor::getGPUBuffer() const noexcept {
  return pointer.toGPUBuffer();
}

Tensor::Tensor(const std::vector<float> &data, const std::vector<size_t> &shape)
    : pointer(data), shape(shape) {
  size_t size = getSize();
  if (size == 0) {
    throw std::invalid_argument("Invalid shape");
  }
  if (data.size() != size) {
    throw std::invalid_argument("Invalid data");
  }
}

Tensor::Tensor(const std::vector<size_t> &shape)
    : pointer(getSize(shape)), shape(shape) {
  if (getSize() == 0) {
    throw std::invalid_argument("Invalid shape");
  }
}
std::vector<size_t> Tensor::getShape() const noexcept { return shape; }
Tensor::Tensor(const std::vector<float> &data)
    : Tensor(data, std::vector<size_t>(1, data.size())) {}

size_t Tensor::getSize() const {
  if (shape.size() == 0) {
    return 0;
  }
  return getSize(shape);
}

size_t Tensor::getSize(const std::vector<size_t> &shape) {
  size_t size = 1;
  for (size_t dim : shape) {
    size *= dim;
  }
  return size;
}

std::vector<float> Tensor::getData() const noexcept {
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

Tensor Tensor::copy() const {
  gpu::gpu_mem_32f buffer = getGPUBuffer();
  Tensor newTensor(shape);
  gpu::gpu_mem_32f newbuffer = newTensor.getGPUBuffer();
  buffer.copyTo(newbuffer, getSize());
  return newTensor;
}

std::vector<size_t> Tensor::broadcast(const std::vector<size_t> &shape1,
                                      const std::vector<size_t> &shape2) {
  std::vector<size_t> a;
  std::vector<size_t> b;
  if (shape1.size() >= shape2.size()) {
    a = shape1;
    b = shape2;
  } else {
    a = shape2;
    b = shape1;
  }

  while (b.size() < a.size()) {
    // this is really bad
    b.insert(b.begin(), 1);
  }

  for (size_t i = 0; i < b.size(); ++i) {
    if (b[i] == 1) {
      b[i] = a[i];
    }
  }

  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] == 1) {
      a[i] = b[i];
    } else if (a[i] != b[i]) {
      throw std::runtime_error("Can't broadcats this vectors");
    }
  }

  return a;
}

Tensor concat(const std::vector<Tensor> &tensors) {
  if (!tensors.size()) {
    throw std::invalid_argument("Tensors is empty!");
  }
  std::vector<size_t> shape = tensors[0].getShape();
  for (size_t i = 1; i < tensors.size(); ++i) {
    if (tensors[i].getShape() != shape) {
      throw std::invalid_argument("Tensors have different shape!");
    }
  }
  shape.insert(shape.begin(), tensors.size());
  Tensor result(shape);
  gpu::gpu_mem_32f buffer = result.getGPUBuffer();
  size_t tensorSize = tensors[0].getSize();
  for (size_t i = 0; i < tensors.size(); ++i) {
    tensors[i].getGPUBuffer().copyToN(buffer, tensorSize);
  }

  throw std::runtime_error("not imlpemented");
}

void printTensorPart(std::ostream &stream, float *data, size_t dataLen,
                     size_t *shape, size_t shapeLen);

std::ostream &operator<<(std::ostream &stream, const Tensor &tensor) {
  // TODO fix this
  std::vector<float> data = tensor.getData();
  std::vector<size_t> shape = tensor.getShape();
  printTensorPart(stream, data.data(), data.size(), shape.data(), shape.size());
  return stream;
}

void printTensorPart(std::ostream &stream, float *data, size_t dataLen,
                     size_t *shape, size_t shapeLen) {
  if (shapeLen == 0) {
    stream << data[0];
    return;
  }
  stream << "[";
  for (size_t i = 0; i < shape[0]; ++i) {
    printTensorPart(stream, data + i, i * dataLen / shape[0], shape + 1,
                    shapeLen - 1);
    if (i < shape[0] - 1) {
      stream << ", ";
    }
  }
  stream << "]" << std::endl;
}

} // namespace NSTTF
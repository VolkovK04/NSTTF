#pragma once

#include <iostream>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/context.h>
#include <vector>

namespace NSTTF {
class AbstractDataPointer {
public:
  AbstractDataPointer() = default;
  virtual ~AbstractDataPointer() = default;
  virtual std::vector<float> toVector() const = 0;
  virtual gpu::gpu_mem_32f toGPUBuffer() const = 0;
};

class GPUPointer : AbstractDataPointer {
private:
  gpu::gpu_mem_32f pointer;

public:
  GPUPointer() = default;
  explicit GPUPointer(gpu::gpu_mem_32f &pointer);
  explicit GPUPointer(const std::vector<float> &vector);
  explicit GPUPointer(size_t size);
  std::vector<float> toVector() const;
  gpu::gpu_mem_32f toGPUBuffer() const;
};

class RAMPointer : AbstractDataPointer {
private:
  std::vector<float> pointer;

public:
  explicit RAMPointer(std::vector<float> &pointer);
  std::vector<float> toVector();
  gpu::gpu_mem_32f toGPUBuffer();
};

class Tensor {
private:
  GPUPointer pointer;
  std::vector<size_t> shape;

  static size_t getSize(const std::vector<size_t> &shape);

public:
  Tensor() = default;

  Tensor(const std::vector<float> &data, const std::vector<size_t> &shape);

  explicit Tensor(const std::vector<size_t> &shape);

  explicit Tensor(const std::vector<float> &vector);

  gpu::gpu_mem_32f getGPUBuffer() const;

  std::vector<size_t> getShape() const;

  std::vector<float> getData() const;

  void reshape(const std::vector<size_t> &newShape);

  size_t getSize() const;
  ~Tensor() = default;
};

} // namespace NSTTF
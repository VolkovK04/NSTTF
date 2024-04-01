#pragma once

#include <iostream>
#include <libgpu/shared_device_buffer.h>
#include <vector>

namespace NSTTF {
class AbstractDataPointer {
  public:
    AbstractDataPointer() = default;
    virtual ~AbstractDataPointer() = default;
    virtual std::vector<float> toVector();
    virtual gpu::gpu_mem_32f toGPUBuffer();
};

class Tensor {
  private:
    GPUPointer pointer;
    const std::vector<size_t> shape;

  public:
    Tensor() = default;

    Tensor(GPUPointer pointer);

    Tensor(const std::vector<size_t>& shape);

    Tensor(const std::vector<float>& vector);

    gpu::gpu_mem_32f getGPUBuffer();

    std::vector<size_t> getShape();
    ~Tensor() = default;

};

class GPUPointer : AbstractDataPointer {
  private:
    gpu::gpu_mem_32f pointer;

  public:
    GPUPointer(gpu::gpu_mem_32f& pointer);
    GPUPointer(const std::vector<float>& vector);
    GPUPointer::GPUPointer(size_t size);
    std::vector<float> toVector();
    gpu::gpu_mem_32f toGPUBuffer();
};

class RAMPointer : AbstractDataPointer {
  private:
    std::vector<float> pointer;

  public:
    RAMPointer(std::vector<float>& pointer);
    std::vector<float> toVector();
    gpu::gpu_mem_32f toGPUBuffer();
};
} // namespace NSTTF
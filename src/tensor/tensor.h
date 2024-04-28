#pragma once

#include <iostream>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <unordered_map>
#include <vector>
#include <unordered_map>
#include <string>

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

  Tensor copy() const;

  size_t getSize() const;
  ~Tensor() = default;
};

class TensorStack {
private:
  std::vector<Tensor> tensors;

public:
  TensorStack() = default;
  explicit TensorStack(std::vector<Tensor> tensors);

  void append(const Tensor &tensor);
  void insert(const Tensor &tensor);
  void remove(size_t index);

  size_t count() const;
  Tensor toTensor() const;

  const Tensor &operator[](size_t index) const;
  // TODO iterators
};

class TensorMap_ {
  private:
    std::unordered_map<std::string, Tensor> data;
  public:
    TensorMap_() = default;
    explicit TensorMap_(const std::unordered_map<std::string, Tensor>& data) : data(data) {}
    void insert(std::string label, const Tensor& tensor);
    void remove(std::string label);
    
    size_t count() const;

    const Tensor& operator[](std::string label) const;
};

typedef std::unordered_map<std::string, Tensor> TensorMap;
// TODO: overload sum for this map
} // namespace NSTTF
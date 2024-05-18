#pragma once

#include <tensor/tensor.h>

namespace NSTTF {
class DataLoader {
public:
  virtual const TensorMap &operator[](size_t index) const = 0;
  virtual size_t size() const = 0;
  virtual ~DataLoader() = default;
};

class MNIST_DataLoader : public DataLoader {
public:
  MNIST_DataLoader();
  size_t size() const override;
  const TensorMap &operator[](size_t index) const override;
};

} // namespace NSTTF
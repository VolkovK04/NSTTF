#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tensor/tensor.h>
#include <vector>

namespace NSTTF {

struct MNIST_dataset {
  std::vector<uint8_t> images;
  std::vector<uint8_t> labels;
};

class DataLoader {
public:
  virtual TensorMap operator[](size_t index) const = 0;
  virtual size_t size() const = 0;
};

class MNIST_DataLoader : public DataLoader {
public:
  MNIST_DataLoader(const std::string &type);
  ~MNIST_DataLoader() { delete dataset; }

  // TensorMap{{"data", data_tensor}, {"label", lable_tensor}};
  TensorMap operator[](size_t index) const override;
  size_t size() const override;

  std::vector<uint8_t> get_images() { return dataset->images; }
  std::vector<uint8_t> get_labels() { return dataset->labels; }

private:
  size_t dataset_size;
  MNIST_dataset *dataset = new MNIST_dataset();

  uint32_t read_header(const std::vector<char> &buffer, size_t position);
  std::vector<char> read_mnist_file(const std::string &name, uint32_t key);
  std::vector<uint8_t> read_mnist_image_file(const std::string &name);
  std::vector<uint8_t> read_mnist_label_file(const std::string &name);
};

} // namespace NSTTF
#pragma once
#include <filesystem>
#include <fstream>
#include <libutils/misc.h>
#include <string>
#include <tensor/tensor.h>
#include <unordered_map>
#include <vector>

namespace NSTTF {

class Function {
public:
  Function() = default;
  Function(const Function &) = delete;
  Function &operator=(const Function &) = delete;

  Function(Function &&) = default;
  Function &operator=(Function &&) = default;


  virtual std::vector<Tensor>
  compute(const std::vector<Tensor> &inputs) const = 0;
  virtual Tensor derivative(const std::vector<Tensor> &inputs,
                            size_t inputIndex, size_t outputIndex,
                            Tensor grad) const = 0;
};

bool init();
void checkNumOfTensors(const std::vector<Tensor> &tensors, size_t num);
void checkShape(Tensor &arg1, Tensor &arg2);
std::vector<char> clToCharVector(const std::string &clFilename);

extern std::unordered_map<std::string, std::shared_ptr<Function>> functions_;
extern std::unordered_map<std::string, ocl::Kernel> kernels;

class UnaryMinus : public Function {
  friend bool init();

public:
  UnaryMinus() = default;
  std::vector<Tensor> compute(const std::vector<Tensor> &inputs) const override;
  Tensor derivative(const std::vector<Tensor> &inputs, size_t inputIndex,
                    size_t outputIndex, Tensor grad) const override;
};

class Sum : public Function {
  friend bool init();

public:
  Sum() = default;
  std::vector<Tensor> compute(const std::vector<Tensor> &inputs) const override;
  Tensor derivative(const std::vector<Tensor> &inputs, size_t inputIndex,
                    size_t outputIndex, Tensor grad) const override;
};

class Subtration : public Function {
  friend bool init();

public:
  Subtration() = default;
  std::vector<Tensor> compute(const std::vector<Tensor> &inputs) const override;
  Tensor derivative(const std::vector<Tensor> &inputs, size_t inputIndex,
                    size_t outputIndex, Tensor grad) const override;
};

class Multiplication : public Function {
  friend bool init();

public:
  Multiplication() = default;
  std::vector<Tensor> compute(const std::vector<Tensor> &inputs) const override;
  Tensor derivative(const std::vector<Tensor> &inputs, size_t inputIndex,
                    size_t outputIndex, Tensor grad) const override;
};

class MatrixMultiplication : public Function {
  friend bool init();

public:
  MatrixMultiplication() = default;
  std::vector<Tensor> compute(const std::vector<Tensor> &inputs) const override;
  Tensor derivative(const std::vector<Tensor> &inputs, size_t inputIndex,
                    size_t outputIndex, Tensor grad) const override;
};

class MatrixTranspose : public Function {
  friend bool init();

public:
  MatrixTranspose() = default;
  std::vector<Tensor> compute(const std::vector<Tensor> &inputs) const override;
  Tensor derivative(const std::vector<Tensor> &inputs, size_t inputIndex,
                    size_t outputIndex, Tensor grad) const override;
};

} // namespace NSTTF
#pragma once
#include <compiler/instruction.h>
#include <libutils/misc.h>
#include <string>
#include <tensor/tensor.h>
#include <unordered_map>
#include <vector>

namespace NSTTF {

class Function;

extern std::unordered_map<std::string, std::shared_ptr<Function>> functions;
extern std::unordered_map<std::string, ocl::Kernel> kernels;

class Function {
public:
  Function() = default;
  virtual ~Function() = default;
  // why??
  Function(const Function &) = delete;
  Function &operator=(const Function &) = delete;

  Function(Function &&) = default;
  Function &operator=(Function &&) = default;

  virtual Tensor compute(const std::vector<Tensor> &inputs) const = 0;
  virtual std::vector<AbstractInstruction *>
  derivative(const std::vector<std::string> &inputs, size_t inputIndex,
             const std::string &grad, const std::string &resultName) const = 0;
};

void init();
void checkNumOfTensors(const std::vector<Tensor> &tensors, size_t num);
void checkShape(Tensor &arg1, Tensor &arg2);
std::vector<char> clToCharVector(const std::string &clFilename);

class UnaryMinus : public Function {
  friend void init();

public:
  UnaryMinus() = default;
  Tensor compute(const std::vector<Tensor> &inputs) const override;
  std::vector<AbstractInstruction *>
  derivative(const std::vector<std::string> &inputs, size_t inputIndex,
             const std::string &grad,
             const std::string &resultName) const override;
};

class Sum : public Function {
  friend void init();

public:
  Sum() = default;
  Tensor compute(const std::vector<Tensor> &inputs) const override;
  std::vector<AbstractInstruction *>
  derivative(const std::vector<std::string> &inputs, size_t inputIndex,
             const std::string &grad,
             const std::string &resultName) const override;
};

class Subtraction : public Function {
  friend void init();

public:
  Subtraction() = default;
  Tensor compute(const std::vector<Tensor> &inputs) const override;
  std::vector<AbstractInstruction *>
  derivative(const std::vector<std::string> &inputs, size_t inputIndex,
             const std::string &grad,
             const std::string &resultName) const override;
};

class Multiplication : public Function {
  friend void init();

public:
  Multiplication() = default;
  Tensor compute(const std::vector<Tensor> &inputs) const override;
  std::vector<AbstractInstruction *>
  derivative(const std::vector<std::string> &inputs, size_t inputIndex,
             const std::string &grad,
             const std::string &resultName) const override;
};

class MatrixMultiplication : public Function {
  friend void init();

public:
  MatrixMultiplication() = default;
  Tensor compute(const std::vector<Tensor> &inputs) const override;
  std::vector<AbstractInstruction *>
  derivative(const std::vector<std::string> &inputs, size_t inputIndex,
             const std::string &grad,
             const std::string &resultName) const override;
};

class MatrixTranspose : public Function {
  friend void init();

public:
  MatrixTranspose() = default;
  // ~MatrixTranspose() noexcept = default;
  Tensor compute(const std::vector<Tensor> &inputs) const override;
  std::vector<AbstractInstruction *>
  derivative(const std::vector<std::string> &inputs, size_t inputIndex,
             const std::string &grad,
             const std::string &resultName) const override;
};

class ReduceSum : public Function {
  friend void init();

public:
  ReduceSum() = default;
  Tensor compute(const std::vector<Tensor> &inputs) const override;
  std::vector<AbstractInstruction *>
  derivative(const std::vector<std::string> &inputs, size_t inputIndex,
             const std::string &grad,
             const std::string &resultName) const override;
};

class CrossEntropy : public Function {
  friend void init();

public:
  CrossEntropy() = default;
  Tensor compute(const std::vector<Tensor> &inputs) const override;
  std::vector<AbstractInstruction *>
  derivative(const std::vector<std::string> &inputs, size_t inputIndex,
             const std::string &grad,
             const std::string &resultName) const override;
};

} // namespace NSTTF
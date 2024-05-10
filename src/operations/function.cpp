#include "function.h"
#include <filesystem>
#include <fstream>

namespace NSTTF {

std::unordered_map<std::string, std::shared_ptr<Function>> initFunctions();

std::unordered_map<std::string, std::shared_ptr<Function>> functions =
    initFunctions();
std::unordered_map<std::string, ocl::Kernel> kernels;

unsigned int workGroupSize_ = 128;

ocl::Kernel prepareKernel(const std::string &clFilename,
                          const std::string &methodName) {
  std::vector<char> source = clToCharVector(clFilename);
  ocl::Kernel kernel(source.data(), source.size(), methodName);
  kernel.compile();
  return std::move(kernel);
}

std::unordered_map<std::string, std::shared_ptr<Function>> initFunctions() {
  std::unordered_map<std::string, std::shared_ptr<Function>> functions_;
  functions_.insert({"unary_minus", std::make_shared<UnaryMinus>()});
  functions_.insert({"subtraction", std::make_shared<Subtraction>()});
  functions_.insert({"multiplication", std::make_shared<Multiplication>()});
  functions_.insert({"sum", std::make_shared<Sum>()});
  functions_.insert(
      {"matrix_multiplication", std::make_shared<MatrixMultiplication>()});
  functions_.insert({"matrix_transpose", std::make_shared<MatrixTranspose>()});
  return std::move(functions_);
}

bool initflag = false;
void init() {
  if (initflag) {
    return;
  }
  initflag = true;

  ocl::Kernel _unaryMinus =
      prepareKernel("src/cl/unary_minus.cl", "unary_minus");
  kernels.insert({"unary_minus", _unaryMinus});

  ocl::Kernel _subtraction =
      prepareKernel("src/cl/subtraction.cl", "subtraction");
  kernels.insert({"subtraction", _subtraction});

  ocl::Kernel _multiplication =
      prepareKernel("src/cl/multiplication.cl", "multiplication");
  kernels.insert({"multiplication", _multiplication});

  ocl::Kernel _sum = prepareKernel("src/cl/sum.cl", "sum");
  kernels.insert({"sum", _sum});

  ocl::Kernel _matrix_multiplication = prepareKernel(
      "src/cl/matrix_multiplication.cl", "matrix_multiplication_updated");
  kernels.insert({"matrix_multiplication", _matrix_multiplication});

  ocl::Kernel _matrix_transpose =
      prepareKernel("src/cl/matrix_transpose.cl", "matrix_transpose");
  kernels.insert({"matrix_transpose", _matrix_transpose});
}

std::vector<char> clToCharVector(const std::string &clFilename) {
  std::filesystem::path sourcePath(_PROJECT_SOURCE_DIR);
  sourcePath.append(clFilename);
  std::ifstream file(sourcePath, std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("Can't open cl file. Path: " +
                             sourcePath.string());
  }
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  std::vector<char> result(size);
  file.seekg(0, std::ios::beg);
  file.read(result.data(), size);
  file.close();
  return std::move(result);
}

void checkShape(Tensor &arg1, Tensor &arg2) {
  if (arg1.getShape() != arg2.getShape()) {
    throw std::runtime_error("Different shape");
  }
}

void checkNumOfTensors(const std::vector<Tensor> &tensors, size_t num) {
  if (tensors.size() < num) {
    throw std::runtime_error("Not enought tensors");
  } else if (tensors.size() > num) {
    throw std::runtime_error("Too many tensors");
  }
}

Tensor Sum::compute(const std::vector<Tensor> &inputs) const {
  checkNumOfTensors(inputs, 2);

  Tensor arg1 = inputs[0];
  Tensor arg2 = inputs[1];
  checkShape(arg1, arg2);

  Tensor res(arg1.getShape());
  unsigned int n = arg1.getSize();
  unsigned int global_work_size =
      (n + workGroupSize_ - 1) / workGroupSize_ * workGroupSize_;
  kernels.at("sum").exec(gpu::WorkSize(workGroupSize_, global_work_size),
                         arg1.getGPUBuffer(), arg2.getGPUBuffer(),
                         res.getGPUBuffer(), n);
  return res;
}

std::vector<AbstractInstruction *>
Sum::derivative(const std::vector<std::string> &inputs, size_t inputIndex,
                const std::string &grad, const std::string &resultName) const {
  if (inputIndex > 2) {
    throw std::out_of_range("inputIndex out of range");
  }
  Instruction *inst =
      new Instruction("copy", {grad}, resultName); // d(x+y)/dx * grad = grad
  std::vector<AbstractInstruction *> result = {inst};
  return std::move(result);
}

Tensor Multiplication::compute(const std::vector<Tensor> &inputs) const {
  checkNumOfTensors(inputs, 2);

  Tensor arg1 = inputs[0];
  Tensor arg2 = inputs[1];

  checkShape(arg1, arg2);

  Tensor res(arg1.getShape());
  unsigned int n = arg1.getSize();
  unsigned int global_work_size =
      (n + workGroupSize_ - 1) / workGroupSize_ * workGroupSize_;
  kernels.at("multiplication")
      .exec(gpu::WorkSize(workGroupSize_, global_work_size),
            arg1.getGPUBuffer(), arg2.getGPUBuffer(), res.getGPUBuffer(), n);
  return res;
}

std::vector<AbstractInstruction *>
Multiplication::derivative(const std::vector<std::string> &inputs,
                           size_t inputIndex, const std::string &grad,
                           const std::string &resultName) const {
  if (inputIndex >= 2) {
    throw std::out_of_range("inputIndex or out of range");
  }
  std::vector<AbstractInstruction *> result;
  if (inputIndex == 0) {
    result.push_back(
        new Instruction("multiplication", {grad, inputs[1]},
                        resultName)); // d(x*y)/dx * grad = y * grad
  } else {
    result.push_back(
        new Instruction("multiplication", {grad, inputs[0]},
                        resultName)); // d(x*y)/dy * grad = x * grad
  }

  return std::move(result);
}

Tensor Subtraction::compute(const std::vector<Tensor> &inputs) const {
  checkNumOfTensors(inputs, 2);

  Tensor arg1 = inputs[0];
  Tensor arg2 = inputs[1];

  checkShape(arg1, arg2);

  Tensor res(arg1.getShape());
  unsigned int n = arg1.getSize();
  unsigned int global_work_size =
      (n + workGroupSize_ - 1) / workGroupSize_ * workGroupSize_;
  kernels.at("subtraction")
      .exec(gpu::WorkSize(workGroupSize_, global_work_size),
            arg1.getGPUBuffer(), arg2.getGPUBuffer(), res.getGPUBuffer(), n);
  return res;
}

std::vector<AbstractInstruction *>
Subtraction::derivative(const std::vector<std::string> &inputs,
                        size_t inputIndex, const std::string &grad,
                        const std::string &resultName) const {
  if (inputIndex >= 2) {
    throw std::out_of_range("input index out of range");
  }

  std::vector<AbstractInstruction *> res;

  if (inputIndex == 0) {
    res.push_back(new Instruction("copy", {grad}, resultName));
  } else {
    res.push_back(new Instruction("unary_minus", {grad}, resultName));
  }
  return std::move(res);
}

Tensor UnaryMinus::compute(const std::vector<Tensor> &inputs) const {
  checkNumOfTensors(inputs, 1);
  Tensor arg1 = inputs[0];
  Tensor res(arg1.getShape());
  unsigned int n = arg1.getSize();
  unsigned int global_work_size =
      (n + workGroupSize_ - 1) / workGroupSize_ * workGroupSize_;
  kernels.at("unary_minus")
      .exec(gpu::WorkSize(workGroupSize_, global_work_size),
            arg1.getGPUBuffer(), res.getGPUBuffer(), n);
  return res;
}

std::vector<AbstractInstruction *>
UnaryMinus::derivative(const std::vector<std::string> &inputs,
                       size_t inputIndex, const std::string &grad,
                       const std::string &resultName) const {
  if (inputIndex != 0) {
    throw std::out_of_range("input index out of range");
  }
  std::vector<AbstractInstruction *> res;
  res.push_back(new Instruction("unary_minus", {grad}, resultName));
  return std::move(res);
}
Tensor MatrixMultiplication::compute(const std::vector<Tensor> &inputs) const {
  checkNumOfTensors(inputs, 2);

  Tensor arg1 = inputs[0];
  Tensor arg2 = inputs[1];

  std::vector<size_t> arg1Shape = arg1.getShape();
  std::vector<size_t> arg2Shape = arg2.getShape();

  size_t arg1Col = arg1Shape[0], arg1Rows = arg1Shape[1];

  size_t arg2Col = arg2Shape[0], arg2Rows = arg2Shape[1];

  if (arg1Rows != arg2Col) {
    throw std::runtime_error("Wrong matrix shape");
  }

  arg1Shape.pop_back();
  arg1Shape.push_back(arg2Rows);

  Tensor res(arg1Shape);
  unsigned int M = arg1Col;
  unsigned int K = arg2Col;
  unsigned int N = arg2Rows;
  unsigned int x_work_group_size = 16;
  unsigned int y_work_group_size = 16;
  unsigned int x_work_size =
      (M + x_work_group_size - 1) / x_work_group_size * x_work_group_size;
  unsigned int y_work_size =
      (N + y_work_group_size - 1) / y_work_group_size * y_work_group_size;
  kernels.at("matrix_multiplication")
      .exec(gpu::WorkSize(gpu::WorkSize(x_work_group_size, y_work_group_size,
                                        x_work_size, y_work_size)),
            arg1.getGPUBuffer(), arg2.getGPUBuffer(), res.getGPUBuffer(), M, K,
            N);
  return res;
}

std::vector<AbstractInstruction *>
MatrixMultiplication::derivative(const std::vector<std::string> &inputs,
                                 size_t inputIndex, const std::string &grad,
                                 const std::string &resultName) const {
  if (inputIndex >= 2) {
    throw std::out_of_range("input index out of range");
  }
  std::vector<AbstractInstruction *> result;
  if (inputIndex == 0) {
    result.push_back(new Instruction("matrix_transpose", {inputs[1]}, "~tmp"));
    result.push_back(
        new Instruction("matrix_multiplication", {grad, "~tmp"}, resultName));
    // inputs[0].shape ={n, m}
  } else if (inputIndex == 1) {
    result.push_back(new Instruction("matrix_transpose", {inputs[0]}, "~tmp"));
    result.push_back(
        new Instruction("matrix_multiplication", {"~tmp", grad}, resultName));
    // inputs[1].shape = {m, k}     grad.shape = {n, k}
  }
  return std::move(result);
}
Tensor MatrixTranspose::compute(const std::vector<Tensor> &inputs) const {
  checkNumOfTensors(inputs, 1);

  Tensor arg = inputs[0];

  std::vector<size_t> baseShape = arg.getShape();

  size_t argCol = baseShape[1], argRows = baseShape[0];

  baseShape[0] = argCol;
  baseShape[1] = argRows;

  Tensor res(baseShape);
  unsigned int M = argRows;
  unsigned int K = argCol;
  unsigned int x_work_group_size = 16;
  unsigned int y_work_group_size = 16;
  unsigned int x_work_size =
      (M + x_work_group_size - 1) / x_work_group_size * x_work_group_size;
  unsigned int y_work_size =
      (K + y_work_group_size - 1) / y_work_group_size * y_work_group_size;
  kernels.at("matrix_transpose")
      .exec(gpu::WorkSize(x_work_group_size, y_work_group_size, x_work_size,
                          y_work_size),
            arg.getGPUBuffer(), res.getGPUBuffer(), M, K);
  return res;
}
std::vector<AbstractInstruction *>
MatrixTranspose::derivative(const std::vector<std::string> &inputs,
                            size_t inputIndex, const std::string &grad,
                            const std::string &resultName) const {
  if (inputIndex != 0) {
    throw std::out_of_range("input index out of range");
  }
  std::vector<AbstractInstruction *> res;
  res.push_back(new Instruction("matrix_transpose", {grad}, resultName));
  return std::move(res);
}
} // namespace NSTTF
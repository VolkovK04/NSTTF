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
  functions_.insert({"subtraction", std::make_shared<Subtration>()});
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

std::vector<Tensor> Sum::compute(const std::vector<Tensor> &inputs) const {
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
  std::vector<Tensor> result{res};
  return std::move(result);
}

Tensor Sum::derivative(const std::vector<Tensor> &inputs, size_t inputIndex,
                       size_t outputIndex, Tensor grad) const {
  return grad; // d(x+y)/dx * grad = grad
}

std::vector<Tensor>
Multiplication::compute(const std::vector<Tensor> &inputs) const {
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

  std::vector<Tensor> result{res};
  return std::move(result);
}

Tensor Multiplication::derivative(const std::vector<Tensor> &inputs,
                                  size_t inputIndex, size_t outputIndex,
                                  Tensor grad) const {
  if (inputIndex >= 2 || outputIndex >= 1) {
    throw std::out_of_range("inputIndex or outputIndex out of range");
  }
  if (inputIndex == 0) {
    return functions.at("multiplication")
        ->compute({grad, inputs[1]})[0]; // d(x*y)/dx * grad = y * grad
  } else {
    return functions.at("multiplication")
        ->compute({inputs[0], grad})[0]; // d(x*y)/dy * grad = x * grad
  }
}

std::vector<Tensor>
Subtration::compute(const std::vector<Tensor> &inputs) const {
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
  std::vector<Tensor> result{res};
  return std::move(result);
}

Tensor Subtration::derivative(const std::vector<Tensor> &inputs,
                              size_t inputIndex, size_t outputIndex,
                              Tensor grad) const {
  if (inputIndex >= 2 || outputIndex >= 1) {
    throw std::out_of_range("inputIndex or outputIndex out of range");
  }
  if (inputIndex == 0)
    return grad; // d(x-y)/dx * grad = grad
  else {
    return functions.at("minus")->compute({grad})[0];
  }
}

std::vector<Tensor>
UnaryMinus::compute(const std::vector<Tensor> &inputs) const {
  checkNumOfTensors(inputs, 1);
  Tensor arg1 = inputs[0];
  Tensor res(arg1.getShape());
  unsigned int n = arg1.getSize();
  unsigned int global_work_size =
      (n + workGroupSize_ - 1) / workGroupSize_ * workGroupSize_;
  kernels.at("unary_minus")
      .exec(gpu::WorkSize(workGroupSize_, global_work_size),
            arg1.getGPUBuffer(), res.getGPUBuffer(), n);
  std::vector<Tensor> result{res};
  return std::move(result);
}

Tensor UnaryMinus::derivative(const std::vector<Tensor> &inputs,
                              size_t inputIndex, size_t outputIndex,
                              Tensor grad) const {
  if (inputIndex == 0) {
    return functions.at("unary_minus")->compute({grad})[0];
  }
}
std::vector<Tensor>
MatrixMultiplication::compute(const std::vector<Tensor> &inputs) const {
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
  return std::move(std::vector<Tensor>{res});
}

Tensor MatrixMultiplication::derivative(const std::vector<Tensor> &inputs,
                                        size_t inputIndex, size_t outputIndex,
                                        Tensor grad) const {
  if (inputIndex == 0) {
    // std::vector<Instruction> instructions;
    // instructions.push_back(Instruction("matrix_transpose"), {inputs[1].getName()}, {"tmp"});
    // instructions.push_back(Instruction("matrix_multiplication"), {inputs[0].getName(), "tmp"}, {output[0].getName()});
    // return instructions;
    Tensor newInput = functions.at("matrix_transpose")->compute({inputs[1]})[0];
    return functions.at("matrix_multiplication")
        ->compute({grad, newInput})[0]; // inputs[0].shape ={n, m}
  } else if (inputIndex == 1) {
    Tensor newInput = functions.at("matrix_transpose")->compute({inputs[0]})[0];
    return functions.at("matrix_multiplication")
        ->compute({newInput,
                   grad})[0]; // inputs[1].shape = {m, k}     grad.shape = {n, k}
  }
}
std::vector<Tensor>
MatrixTranspose::compute(const std::vector<Tensor> &inputs) const {
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
  std::vector<Tensor> result{res};
  return std::move(result);
}
Tensor MatrixTranspose::derivative(const std::vector<Tensor> &inputs,
                                   size_t inputIndex, size_t outputIndex,
                                   Tensor grad) const {
  // TODO
  // return Tensor();
  throw std::runtime_error("Not implemented yet");
}
} // namespace NSTTF
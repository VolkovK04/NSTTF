#include "function.h"

namespace NSTTF {

std::unordered_map<std::string, std::shared_ptr<Function>> functions_;
std::unordered_map<std::string, ocl::Kernel> kernels;
bool init_flag = true;
unsigned int workGroupSize_ = 128;

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
    // exeption out of range
    throw std::out_of_range("inputIndex or outputIndex out of range");
  }
  if (inputIndex == 0) {

  }
  // return functions["mult"].compute(
  //     {grad, inputs[1]}); // d(x*y)/dx * grad = y * grad
  else {
    // return functions["mult"].compute(
    //     {grad, inputs[0]}); // d(x*y)/dy * grad = x * grad
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
    // exeption out of range
    throw std::out_of_range("inputIndex or outputIndex out of range");
  }
  if (inputIndex == 0)
    return grad; // d(x-y)/dx * grad = grad
  else {
    // functions["minus"].compute(grad);
    // return UnarMinus::compute(grad); // function call
  }
}

ocl::Kernel prepareKernel(const std::string &clFilename) {
  std::vector<char> source = clToCharVector(clFilename);
  ocl::Kernel kernel(source.data(), source.size(), clFilename);
  kernel.compile();
  
  return kernel;
}

void init() {
  if (init_flag) {
    return;
  }
  init_flag = false;

  ocl::Kernel _unaryMinus = prepareKernel("src/cl/unary_minus.cl");
  kernels.insert({"unary_minus", _unaryMinus});

  ocl::Kernel _subtraction = prepareKernel("src/cl/subtraction.cl");
  kernels.insert({"subtraction", _subtraction});

  ocl::Kernel _multiplication = prepareKernel("src/cl/multiplication.cl");
  kernels.insert({"multiplication", _multiplication});

  ocl::Kernel _sum = prepareKernel("src/cl/sum.cl");
  kernels.insert({"sum", _sum});

  ocl::Kernel _matrix_multiplication =
      prepareKernel("src/cl/matrix_multiplication.cl");
  kernels.insert({"matrix_multiplication", _matrix_multiplication});

  ocl::Kernel _matrix_transpose = prepareKernel("src/cl/matrix_transpose.cl");
  kernels.insert({"matrix_transpose", _matrix_transpose});

  functions_.insert({"unary_minus", std::make_shared<UnaryMinus>()});
  functions_.insert({"subtraction", std::make_shared<Subtration>()});
  functions_.insert({"multiplication", std::make_shared<Multiplication>()});
  functions_.insert({"sum", std::make_shared<Sum>()});
  functions_.insert(
      {"matrix_multiplication", std::make_shared<MatrixMultiplication>()});
  functions_.insert({"matrix_transpose", std::make_shared<MatrixTranspose>()});
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
  // TODO
  if (inputIndex == 0) {
    functions_.at("unary_minus")->compute({grad});
    return grad;
  } else {
    // return {0, grad.getShape()};
  }
}
std::vector<Tensor>
MatrixMultiplication::compute(const std::vector<Tensor> &inputs) const {
  return std::vector<Tensor>();
}
Tensor MatrixMultiplication::derivative(const std::vector<Tensor> &inputs,
                                        size_t inputIndex, size_t outputIndex,
                                        Tensor grad) const {
  // TODO
  // return ;
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
}
} // namespace NSTTF
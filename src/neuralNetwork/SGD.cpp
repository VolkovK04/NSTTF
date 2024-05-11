#include "SGD.h"
#include <compiler/compiler.h>
#include <operations/function.h>

namespace NSTTF {

TensorMap scalarMult(TensorMap a, float k);
TensorMap sum(TensorMap a, TensorMap b);

void GD(ComputationGraph &g, TensorMap data, float learningRate,
        const std::vector<std::string> &inputs) {
  Compiler compiler;
  GraphExecutorWG executor = compiler.compileWithGrads(g, inputs);
  // float loss = executor.execute(data)["loss"].getData()[0];
  TensorMap grads = executor.executeGrads();
  data = sum(data, scalarMult(data, -learningRate));
}

TensorMap sum(TensorMap a, TensorMap b) {
  TensorMap result;
  for (auto pair : a) {
    std::string key = pair.first;
    Tensor valueA = pair.second;
    Tensor valueB = b.at(key);
    Tensor resultValue = functions.at("sum")->compute({valueA, valueB});
    result.insert({key, resultValue});
  }
  return result;
}

TensorMap scalarMult(TensorMap a, float k) {
  TensorMap result;
  for (auto pair : a) {
    std::string key = pair.first;
    Tensor valueA = pair.second;
    std::vector dataB(valueA.getSize(), k);
    Tensor valueB(dataB, valueA.getShape());
    Tensor resultValue = functions.at("multiply")->compute({valueA, valueB});
    result.insert({key, resultValue});
  }
  return result;
}
} // namespace NSTTF
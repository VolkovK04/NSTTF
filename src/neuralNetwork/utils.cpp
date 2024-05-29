#include "utils.h"

#include <operations/function.h>
#include <random>

namespace NSTTF {
std::mt19937 generator;

void setSeed(int seed) { generator.seed(seed); }

std::vector<float> generateUniformRandomVector(size_t n, float minValue,
                                               float maxValue) {
  std::vector<float> result(n);

  std::uniform_real_distribution<float> distribution(minValue, maxValue);
  for (size_t i = 0; i < n; ++i) {
    result[i] = distribution(generator);
  }

  return result;
}

size_t getIndexOfMaxElement(const std::vector<float> &vec) {
  if (vec.empty()) {
    throw std::out_of_range("Vector is empty.");
  }

  size_t maxIndex = 0;
  for (size_t i = 1; i < vec.size(); ++i) {
    if (vec[i] > vec[maxIndex]) {
      maxIndex = i;
    }
  }

  return maxIndex;
}

size_t getRandomIndex(size_t n) {
  if (n == 0) {
    throw std::out_of_range("Count = 0");
  }

  std::uniform_int_distribution<size_t> distribution(0, n - 1);

  return distribution(generator);
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
    Tensor resultValue =
        functions.at("multiplication")->compute({valueA, valueB});
    result.insert({key, resultValue});
  }
  return result;
}
} // namespace NSTTF
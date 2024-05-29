#pragma once

#include <tensor/tensor.h>
#include <vector>

namespace NSTTF {
std::vector<float> generateUniformRandomVector(size_t n, float minValue,
                                               float maxValue);

size_t getRandomIndex(size_t n);
TensorMap sum(TensorMap a, TensorMap b);
TensorMap scalarMult(TensorMap a, float k);
size_t getIndexOfMaxElement(const std::vector<float> &vec);
void setSeed(int seed);

} // namespace NSTTF
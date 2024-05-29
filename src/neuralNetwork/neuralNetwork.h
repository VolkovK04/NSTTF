#pragma once

#include <computationGraph/computationGraph.h>
#include <dataLoader/dataLoader.h>
#include <executor/graphExecutor.h>

namespace NSTTF {
class MNIST_pipeline {
private:
  float learningRate = 0;
  MNIST_DataLoader trainDataLoader;
  MNIST_DataLoader testDataLoader;
  GraphExecutorWG executor;
  TensorMap params;
  TensorMap result;

public:
  MNIST_pipeline();
  // return accuracy
  float testing();
  float training(int epochs, bool verbose = false);
  float training(bool verbose = false);
  void forward(Tensor data, Tensor expected);
  void backward();
  void setLearningRate(float newLearningRate);
  // use after forward
  Tensor getPrediction() const;
  // use after forward
  float getLoss() const;
};

} // namespace NSTTF
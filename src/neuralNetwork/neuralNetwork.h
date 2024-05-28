#pragma once

#include <computationGraph/computationGraph.h>
#include <dataLoader/dataLoader.h>
#include <executor/graphExecutor.h>

namespace NSTTF {
class NN {
public:
  virtual void forward(Tensor data) = 0;
  virtual void backward() = 0;
};

class NN_MNIST : public NN {
private:
  float learningRate;
  GraphExecutor executor;
  MNIST_DataLoader trainDataLoader;
  MNIST_DataLoader testDataLoader;
  TensorMap params;
  TensorMap result;

public:
  NN_MNIST();
  void forward(Tensor data) override;
  void backward() override;
  void setLearningRate(float newLearningRate);
  float getLoss() const;
};

} // namespace NSTTF
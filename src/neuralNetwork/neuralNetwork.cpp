#include "neuralNetwork.h"
#include <cassert>
#include <compiler/compiler.h>

#define IMAGE_SIZE 28

namespace NSTTF {
NN_MNIST::NN_MNIST() : trainDataLoader("train"), testDataLoader("test") {
  assert(trainDataLoader.size() > 0);
  assert(testDataLoader.size() > 0);

  // Create ComputationGraph
  ComputationGraph g;
  NodeInterface input = g.AddInputNode("input");
  NodeInterface W = g.AddInputNode("W");
  NodeInterface b = g.AddInputNode("b");
  NodeInterface tmp1 = NodeInterface::MatrixMult(W, input);
  NodeInterface tmp2 = tmp1 + b;

  NodeInterface y = g.AddInputNode("y");
  NodeInterface delta = tmp2 - y;
  NodeInterface deltaSquare = delta * delta;

  NodeInterface loss = g.AddOperationNode("reduce_sum", {deltaSquare});
  loss.setName("loss");

  // Create executor
  Compiler compiler;
  executor = compiler.compileWithGrads(g, {"W", "b"});

  // Init params
  Tensor W_data;
  Tensor b_data;
  params.insert({"W", W_data});
  params.insert({"b", b_data});
}

void NN_MNIST::forward(Tensor input) {
  std::vector<size_t> shape = input.getShape();

  assert(shape.size() == 2);
  assert(shape[0] == IMAGE_SIZE);
  assert(shape[1] == IMAGE_SIZE);

  input.reshape({IMAGE_SIZE * IMAGE_SIZE}); // flatten

  TensorMap inputMap(params);
  inputMap.insert({"input", input});
  result = executor.execute(inputMap);
}

void NN_MNIST::backward() {
  // TODO
}

void NN_MNIST::setLearningRate(float newLearningRate) {
  learningRate = newLearningRate;
}

float NN_MNIST::getLoss() const {
  Tensor lossTensor = result.at("loss");
  std::vector<float> lossData = lossTensor.getData();
  assert(lossData.size() == 1);
  return lossData[0];
}
} // namespace NSTTF
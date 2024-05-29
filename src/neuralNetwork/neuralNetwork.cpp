#include "neuralNetwork.h"
#include "utils.h"
#include <cassert>
#include <compiler/compiler.h>

#define IMAGE_SIZE 28
#define IMAGE_LEN IMAGE_SIZE *IMAGE_SIZE

namespace NSTTF {

GraphExecutorWG createExecutor() {

  // Create ComputationGraph
  ComputationGraph g;
  NodeInterface input = g.AddInputNode("input");
  NodeInterface W = g.AddInputNode("W");
  NodeInterface b = g.AddInputNode("b");
  NodeInterface tmp1 = NodeInterface::MatrixMult(W, input);
  NodeInterface result = tmp1 + b;
  result.setName("result");
  result.setOutput();

  NodeInterface expected = g.AddInputNode("expected");
  NodeInterface delta = result - expected;
  NodeInterface deltaSquare = delta * delta;

  NodeInterface loss = g.AddOperationNode("reduce_sum", {deltaSquare});
  loss.setName("loss");
  loss.setOutput();

  // Create executor
  Compiler compiler;
  return compiler.compileWithGrads(g);
}

MNIST_pipeline::MNIST_pipeline()
    : trainDataLoader("train"), testDataLoader("test"),
      executor(createExecutor()) {
  // Init params
  size_t numNeurons = 10;

  std::vector<float> W_data =
      generateUniformRandomVector(numNeurons * IMAGE_LEN, -1, 1);
  std::vector<float> b_data = generateUniformRandomVector(numNeurons, -1, 1);

  Tensor W_tensor(W_data, {numNeurons, IMAGE_LEN});
  Tensor b_tensor(b_data, {numNeurons, 1});
  params.insert({"W", W_tensor});
  params.insert({"b", b_tensor});
}

float MNIST_pipeline::testing() {
  size_t datasetSize = trainDataLoader.size();
  int matches = 0;
  for (size_t i = 0; i < datasetSize; ++i) {
    TensorMap sample = trainDataLoader[i];
    Tensor input = sample.at("data");
    Tensor expected = sample.at("label");
    expected.reshape({expected.getSize(), 1});
    forward(input, expected);
    std::vector<float> preditcion = getPrediction().getData();
    int predictedNumber = getIndexOfMaxElement(preditcion);
    int expectedNumber = getIndexOfMaxElement(expected.getData());
    if (predictedNumber == expectedNumber) {
      ++matches;
    }
  }
  return (float)matches / datasetSize;
}

float MNIST_pipeline::training(int epochs, bool verbose) {
  size_t datasetSize = trainDataLoader.size();
  int matches = 0;
  for (size_t epochNumber = 0; epochNumber < epochs; ++epochNumber) {
    size_t sampleIndex = getRandomIndex(datasetSize);
    TensorMap sample = trainDataLoader[sampleIndex];
    Tensor input = sample.at("data");
    Tensor expected = sample.at("label");
    expected.reshape({expected.getSize(), 1});
    forward(input, expected);
    float lossValue = getLoss();
    if (verbose) {
      std::cout << "Loss: " << lossValue << std::endl;
    }
    backward();
    std::vector<float> preditcion = getPrediction().getData();
    int predictedNumber = getIndexOfMaxElement(preditcion);
    int expectedNumber = getIndexOfMaxElement(expected.getData());
    if (predictedNumber == expectedNumber) {
      ++matches;
    }
  }
  return (float)matches / epochs;
}

void MNIST_pipeline::forward(Tensor input, Tensor expected) {
  std::vector<size_t> shape = input.getShape();

  assert(shape.size() == 2);
  assert(shape[0] == IMAGE_SIZE);
  assert(shape[1] == IMAGE_SIZE);

  input.reshape({IMAGE_SIZE * IMAGE_SIZE, 1}); // flatten

  TensorMap inputMap(params);
  inputMap.insert({"input", input});
  inputMap.insert({"expected", expected});
  result = executor.execute(inputMap);
}

void MNIST_pipeline::backward() {
  TensorMap grads = executor.executeGrads();
  params = sum(params, scalarMult(params, -learningRate));
}

void MNIST_pipeline::setLearningRate(float newLearningRate) {
  learningRate = newLearningRate;
}

Tensor MNIST_pipeline::getPrediction() const { return result.at("result"); }

float MNIST_pipeline::getLoss() const {
  Tensor lossTensor = result.at("loss");
  std::vector<float> lossData = lossTensor.getData();
  assert(lossData.size() == 1);
  return lossData[0];
}
} // namespace NSTTF
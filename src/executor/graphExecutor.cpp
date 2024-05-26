#include "graphExecutor.h"
#include <operations/function.h>

namespace NSTTF {

GraphExecutor::GraphExecutor(
    const std::vector<AbstractInstruction *> &instructions,
    const std::vector<std::string> &outputs)
    : instructions(instructions), outputs(outputs) {}

GraphExecutor::~GraphExecutor() {
  for (auto instruction : instructions) {
    delete instruction;
  }
}

GraphExecutorWG::~GraphExecutorWG() {
  for (auto instruction : gradInstructions) {
    delete instruction;
  }
}

GraphExecutorWG::GraphExecutorWG(
    const std::vector<AbstractInstruction *> &instructions,
    const std::vector<std::string> &outputs,
    const std::vector<std::string> &inputs,
    const std::vector<AbstractInstruction *> &gradient)
    : GraphExecutor::GraphExecutor(instructions, outputs),
      gradInstructions(gradient), inputs(inputs) {}

TensorMap GraphExecutor::execute(const TensorMap &tensorsMap) {
  calculated = tensorsMap;

  for (AbstractInstruction *abstractInstruction : instructions) {
    Instruction *instruction = dynamic_cast<Instruction *>(abstractInstruction);
    std::string instructionName = instruction->getName();
    if (instructionName == "copy") {
      std::string outputName = instruction->getOutput();
      calculated[outputName] = calculated[instruction->getInputs()[0]];
      continue;
    }

    std::vector<Tensor> tensors;

    std::vector<std::string> inputs = instruction->getInputs();
    for (std::string input : inputs) {
      tensors.push_back(calculated[input]);
    }
    Tensor result = functions.at(instructionName)->compute(tensors);
    std::string outputName = instruction->getOutput();
    calculated[outputName] = result;
  }
  TensorMap outputMap;
  for (std::string output : outputs) {
    std::string name = output;
    outputMap[name] = calculated[name];
  }
  return outputMap;
}

void GraphExecutor::printInstructions(std::ostream &stream) const {
  for (auto instruction : instructions) {
    instruction->printInfo(stream);
  }
}

TensorMap GraphExecutorWG::executeGrads() {
  grads = calculated;
  for (AbstractInstruction *abstractInstruction : gradInstructions) {
    ConstInstruction *constInstruction =
        dynamic_cast<ConstInstruction *>(abstractInstruction);
    if (constInstruction) {
      std::string outputName = constInstruction->getOutput();
      grads[outputName] = constInstruction->getTensor();
      continue;
    }

    ExtendInstruction *extendInstruction =
        dynamic_cast<ExtendInstruction *>(abstractInstruction);

    if (extendInstruction) {
      //
      Tensor reducedTensor = grads[extendInstruction->getReducedTensor()];
      std::vector<size_t> shapeReducedTensor = reducedTensor.getShape();
      Tensor grad = grads[extendInstruction->getInput()];
      std::vector<float> gradData = grad.getData();

      std::vector<size_t> gradShape = grad.getShape();

      size_t vectorSize = shapeReducedTensor[0];
      if (gradShape.size() == 1 && shapeReducedTensor.size() >= 2) {
        vectorSize *= shapeReducedTensor[1];
      }
      for (size_t shapeEl : gradShape) {
        vectorSize *= shapeEl;
      };

      std::vector<float> extendedData(vectorSize);

      // TODO: need to do it smarter or with broadcast idk
      size_t reduceShapeIdx = 0;
      for (size_t i = 0; i < vectorSize; ++i) {
        extendedData[i] = gradData[reduceShapeIdx++];
        if (reduceShapeIdx == gradData.size()) {
          reduceShapeIdx = 0;
        }
      }

      std::string outputName = extendInstruction->getOutput();

      gradShape.insert(gradShape.begin(), shapeReducedTensor[0]);

      grads[outputName] = Tensor{extendedData, shapeReducedTensor};
      continue;
    }

    Instruction *instruction = dynamic_cast<Instruction *>(abstractInstruction);
    if (!instruction) {
      throw std::runtime_error("Unlucky ;("); // TODO
    }

    std::string instructionName = instruction->getName();
    if (instructionName == "copy") {
      std::string outputName = instruction->getOutput();
      grads[outputName] = grads[instruction->getInputs()[0]];
      continue;
    }

    std::vector<Tensor> tensors;
    std::vector<std::string> inputs = instruction->getInputs();
    for (std::string input : inputs) {
      tensors.push_back(grads[input]);
    }
    Tensor result = functions.at(instructionName)->compute(tensors);
    std::string outputName = instruction->getOutput();
    grads[outputName] = result;
  }
  TensorMap resultMap;
  for (std::string input : inputs) {
    std::string name = "~grad_" + input;
    resultMap[name] = grads[name];
  }
  return resultMap;
}

void GraphExecutorWG::printGradInstructions(std::ostream &stream) const {
  for (auto instruction : gradInstructions) {
    instruction->printInfo(stream);
  }
}

const std::vector<AbstractInstruction *> &
GraphExecutorWG::getGradInstructions() const {
  return gradInstructions;
}

} // namespace NSTTF
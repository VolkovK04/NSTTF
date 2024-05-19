#include "instruction.h"

namespace NSTTF {

const std::string &AbstractInstruction::getName() const { return name; }

const std::vector<std::string> &Instruction::getInputs() const {
  return inputNodeNames;
}

const std::string &Instruction::getOutput() const { return outputNodeName; }

void AbstractInstruction::printInfo(std::ostream &stream) const {
  stream << name << ";" << std::endl;
}

void Instruction::printInfo(std::ostream &stream) const {
  stream << outputNodeName << " = " << name << "(";
  size_t n = inputNodeNames.size() - 1;
  for (size_t i = 0; i < n; ++i) {
    stream << inputNodeNames[i] << ", ";
  }
  stream << inputNodeNames[n] << ");" << std::endl;
}

void ConstInstruction::printInfo(std::ostream &stream) const {
  stream << outputNodeName << " = " << tensor << std::endl;
}

} // namespace NSTTF
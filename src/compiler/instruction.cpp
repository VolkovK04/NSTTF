#include "instruction.h"

namespace NSTTF {

const std::string &AbstractInstruction::getName() const { return name; }

const std::vector<std::string> &Instruction::getInputs() const {
  return inputNodeNames;
}

const std::string &Instruction::getOutput() const { return outputNodeName; }

} // namespace NSTTF
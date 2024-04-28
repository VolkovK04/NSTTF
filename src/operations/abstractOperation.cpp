#include "abstractOperation.h"

namespace NSTTF {
const std::string AbstractOperation::getName() { return name; }
AbstractOperation::AbstractOperation(std::string name) : name(name) {}

} // namespace NSTTF
#pragma once

#include <string>
namespace NSTTF {

class AbstractOperation {
public:
  std::string name; // TODO
  AbstractOperation() = default;
  virtual ~AbstractOperation() = default;
};

} // namespace NSTTF
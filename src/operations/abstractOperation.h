#pragma once

#include <string>
namespace NSTTF {

class AbstractOperation {

private:
  std::string name;

public:
  const std::string getName();

  AbstractOperation() = default;
  AbstractOperation(std::string name);
  virtual ~AbstractOperation() = default;
};

} // namespace NSTTF
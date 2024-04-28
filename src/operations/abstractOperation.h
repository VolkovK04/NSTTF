#pragma once

#include <string>
#include <memory>

namespace NSTTF
{

  class AbstractOperation
  {

  protected:
    std::string name;

  public:
    const std::string getName();

    AbstractOperation() = default;
    AbstractOperation(std::string name);

    virtual ~AbstractOperation() = default;
  };

} // namespace NSTTF
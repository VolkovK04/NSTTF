#pragma once

#include <string>
namespace NSTTF {

class AbstractOperation {

  private:
    std::string name;

  public:
    const std::string getName();

    AbstractOperation() = default;
    virtual ~AbstractOperation() = default;
};

} // namespace NSTTF
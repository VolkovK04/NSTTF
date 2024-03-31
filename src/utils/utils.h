#pragma once
#include "../operations/abstractOperation.h"
#include <hash_map>
#include <string>

namespace NSTTF {
namespace utils {
std::hash_map<const std::string, const AbstractOperation *> operationNameMap = {
    // TODO: Add more operations
};
}
} // namespace NSTTF

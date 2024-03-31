#pragma once
#include <hash_map>
#include <string>
#include "../operations/abstractOperation.h"

namespace NSTTF{
    namespace utils
    {
        std::hash_map<const std::string, const AbstractOperation*> operationNameMap = {
            // TODO: Add more operations
        };
    }
}

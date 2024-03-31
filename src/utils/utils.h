#pragma once
#include <map>
#include <string>
#include "../operations/abstractOperation.h"

namespace NSTTF
{
    namespace utils
    {
        std::map<const std::string, const AbstractOperation *> operationNameMap = {
            // TODO: Add more operations
        };
    }
}

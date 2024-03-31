#pragma once

#include <iostream>
#include <vector>
#include "node.h"

namespace NSTTF
{
    class ComputatitonGraph
    {
        std::vector<InputNode> input;
        std::vector<OutputNode> output;

    public:
        ComputatitonGraph() = default;
    };
}
#pragma once

#include <iostream>
#include <vector>

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
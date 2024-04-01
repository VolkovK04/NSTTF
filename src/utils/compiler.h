#pragma once 

#include "graphExecutor.h"
#include "../computationGraph/computationGraph.h"
#include "../computationGraph/node.h"

namespace NSTTF 
{
    class Compiler 
    {
        private: 
        void compute(AbstractNode node);
        

        public:
        GraphExecutor compile(const ComputationGraph& graph);
    };
};
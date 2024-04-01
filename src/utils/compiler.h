#pragma once 

#include "graphExecutor.h"
#include "../computationGraph/computationGraph.h"
#include "../computationGraph/node.h"

namespace NSTTF 
{
    class Compiler 
    {
        private: 
        void get_instruction(AbstractNode* node, std::vector<AbstractNode *> result);
        

        public:
        std::vector<AbstractNode *> get_all_instructions();

        GraphExecutor compile(const ComputationGraph& graph);
    };
};
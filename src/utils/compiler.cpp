#include "compiler.h"
#include <unordered_set>
#include <vector>
#include "../computationGraph/node.h"


namespace NSTTF 
{
    class Compiler 
    {
        GraphExecutor compile(const ComputationGraph& graph)
        {

            std::unordered_set<AbstractNode *> computed;
            for (AbstractNode* input : graph.getInputNodes()) 
            {
                computed.insert(input);
            }
            for (AbstractNode* output : graph.getOutputNodes()) 
            {
                auto previousNodes = output->getPreviousNodes();
                for (auto prevNode : previousNodes) 
                {

                }
            }
            return GraphExecutor();
        }
    };
}
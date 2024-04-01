#include "compiler.h"
#include <unordered_set>
#include <vector>
#include "../computationGraph/node.h"

namespace NSTTF
{
    class Compiler
    {
        std::unordered_set<AbstractNode *> computed;
        std::vector<InputNode *> inputs;

        GraphExecutor compile(const ComputationGraph &graph)
        {
            
            inputs = graph.getInputNodes();
            std::vector<AbstractNode *> instructions = get_all_instructions();


            return GraphExecutor();
        }
        void get_instruction(AbstractNode *node, std::vector<AbstractNode *> result)
        {
            if (computed.count(node))
            {
                return;
            }

            result.push_back(new );
            computed.insert(node);
        }

        std::vector<AbstractNode *> get_all_instructions()
        {
            std::vector<AbstractNode *> result;

            for (AbstractNode * inp : inputs){
                get_instruction(inp, result);
            }

            return result;
        } 
    };

}
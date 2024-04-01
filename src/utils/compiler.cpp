#include "compiler.h"
#include "instruction.h"
#include <unordered_set>
#include <vector>
#include "../computationGraph/node.h"

namespace NSTTF
{
    class Compiler
    {
        std::unordered_set<OperationNode *> computed;
        std::vector<InputNode *> inputs;

        GraphExecutor compile(const ComputationGraph &graph)
        {
            inputs = graph.getInputNodes();
            std::vector<OperationNode *> instructions = get_all_instructions();

            GraphExecutor executor(instructions);

            return executor;
        }

        void get_instruction(OperationNode *node, std::vector<Instruction *> result)
        {
            if (computed.count(node))
            {
                return;
            }

            std::vector<std::string> prevNodes;
            std::vector<std::string> nextNodes;

            for (auto prev : node->getPreviousNodes()){
                prevNodes.push_back(prev->getName());
            }

            for (auto next : node->getNextNodes()){
                nextNodes.push_back(next->getName());
            }

            result.push_back(Instruction(node->getOperation().name, prevNodes, nextNodes));
            computed.insert(node);
        }

        std::vector<Instruction *> get_all_instructions()
        {
            std::vector<OperationNode *> result;

            for (InputNode *inp : inputs)
            {
                get_instruction(inp, result);
            }

            return result;
        }
    };

}
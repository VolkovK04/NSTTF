#include "compiler.h"
#include "../computationGraph/node.h"
#include "instruction.h"
#include <unordered_set>
#include <vector>

namespace NSTTF {
class Compiler {
    std::unordered_set<AbstractNode *> computed;
    std::vector<AbstractNode *> outputs;

    GraphExecutor compile(const ComputationGraph &graph) {
        outputs = graph.getOutputNodes();
        std::vector<Instruction> instructions = get_all_instructions();

        GraphExecutor executor(instructions);

        return executor;
    }

    void get_instruction(AbstractNode *node, std::vector<Instruction> &result) {
        if (computed.count(node)) {
            return;
        }

        OperationNode *operatioNode = dynamic_cast<OperationNode *>(node);
        if (!operatioNode) {
            return;
        }

        std::vector<const std::string> prevNodes;
        std::vector<const std::string> nextNodes;

        for (auto prev : operatioNode->getPreviousNodes()) {
            prevNodes.push_back(prev->getName());
        }

        for (auto next : operatioNode->getNextNodes()) {
            nextNodes.push_back(next->getName());
        }
        Instruction instruction(operatioNode->getOperation().getName(),
                                std::move(nextNodes), std::move(prevNodes));
        result.push_back(instruction);
        computed.insert(operatioNode);
    }

    std::vector<Instruction> get_all_instructions() {
        std::vector<Instruction> result;

        for (AbstractNode *inp : outputs) {
            get_instruction(inp, result);
        }

        return result;
    }
};

} // namespace NSTTF
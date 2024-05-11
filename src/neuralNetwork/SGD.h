#include <executor/graphExecutor.h>
#include <computationGraph/computationGraph.h>

namespace NSTTF {
void GD(ComputationGraph& g, TensorMap data, float learningRate,
        const std::vector<std::string> &inputs);
}
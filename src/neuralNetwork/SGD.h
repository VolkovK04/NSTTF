#include <computationGraph/computationGraph.h>
#include <executor/graphExecutor.h>

namespace NSTTF {
void GD(ComputationGraph &g, TensorMap data, float learningRate,
        const std::vector<std::string> &inputs);
}
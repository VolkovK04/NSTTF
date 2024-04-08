// #include <libgpu/context.h>
// #include <libgpu/shared_device_buffer.h>
// #include <libutils/fast_random.h>
// #include <libutils/misc.h>
// #include <libutils/timer.h>


// #include "cl_build_headers/matrix_multiplication_cl.h"
// #include "cl_build_headers/matrix_transpose_cl.h"
// #include "cl_build_headers/multiplication_cl.h"
// #include "cl_build_headers/subtraction_cl.h"
// #include "cl_build_headers/sum_cl.h"


// #include "computationGraph/computationGraph.h"
// #include "computationGraph/node.h"
#include "utils/compiler.h"
// #include "utils/utils.h"
// #include "utils/graphExecutor.h"
// #include "tensor/tensor.h"

#include <vector>

int main(int argc, char **argv) {
  auto g = NSTTF::ComputationGraph();

  auto a = g.AddInputNode("a");
  auto b = g.AddInputNode("b");
  NSTTF::AbstractOperation oper("sum");

  auto c = g.AddOperationNode(oper, {&a, &b}, "c", true);



  return 0;
}

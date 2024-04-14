
// #include "utils/compiler.h"
#include <CL/cl.h>
#include "utils/functions.h"
#include "tensor/tensor.h"
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
// #include <iostream>
#include <vector>
#include <libutils/misc.h>
// #include <map>

using namespace NSTTF;

gpu::Context context;
gpu::Device device;

int main(int argc, char **argv) {
  // auto g = NSTTF::ComputationGraph();

  // auto a = g.AddInputNode("a");
  // auto b = g.AddInputNode("b");
  // NSTTF::AbstractOperation oper("sum");

  // auto c = g.AddOperationNode(oper, {&a, &b}, "c", true);

  // NSTTF::GraphExecutor executor = NSTTF::Compiler().compile(g);

  // NSTTF::Tensor tensor1(std::vector<float>{1, 2, 3});
  // NSTTF::Tensor tensor2(std::vector<float>{4, 5, 6});

  // std::map<std::string, NSTTF::Tensor> tensors = {{"a", tensor1}, {"b", tensor2}};

  // executor.execute(tensors);
    std::vector<gpu::Device> devices = gpu::enumDevices();

    device = devices[devices.size() - 1];
    std::cout << device.name << std::endl;

    context.init(device.device_id_opencl);
    context.activate();

    
    Tensor a(std::vector<float>{1, 2, 3});
    Tensor b(std::vector<float>{4, 5, 6});
    Tensor c = sum({a, b});
    gpu::gpu_mem_32f buff = c.getGPUBuffer();
    std::vector<float> v(3);
    buff.readN(v.data(), 3);
    std::cout << v[0] << v[1] << v[2] << std::endl;
    // std::cout << "Hello world" << std::endl;
  return 0;
}

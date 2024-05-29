#include "tensor/tensor.h"
#include <CL/cl.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>
#include <neuralNetwork/neuralNetwork.h>
#include <operations/function.h>
#include <vector>

using namespace NSTTF;

gpu::Context context;
gpu::Device device;

int main() {
  // std::vector<gpu::Device> devices = gpu::enumDevices();
  // std::cout << devices.size() << " devices" << std::endl;
  // for (auto device : devices) {
  //   device.printInfo();
  // }
  std::vector<gpu::Device> devices = gpu::enumDevices();

  gpu::Device device = devices[devices.size() - 1];

  context.init(device.device_id_opencl);
  context.activate();

  init();

  MNIST_pipeline nn;
  nn.setLearningRate(0.003);
  for (size_t i = 0; i < 100; ++i) {
    float acc = nn.training(100);
    std::cout << acc << std::endl;
  }
  std::cout << "Final accuracy on tests: " << nn.testing() << std::endl;

  return 0;
}

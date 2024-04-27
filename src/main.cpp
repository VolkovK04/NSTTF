#include "tensor/tensor.h"
#include "utils/functions.h"
#include <CL/cl.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>
#include <vector>

using namespace NSTTF;

gpu::Context context;
gpu::Device device;

int main() 
{

  std::vector<gpu::Device> devices = gpu::enumDevices();
  std::cout << devices.size() << " devices" << std::endl;
  for (auto device: devices) {
    device.printInfo();
  }
  return 0;
}

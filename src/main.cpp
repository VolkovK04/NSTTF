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

  std::vector<  gpu::Device> devices =              gpu::enumDevices();
  device = devices[devices.size()-1];
  std::cout << device.name << std::endl;

  return 0;
}

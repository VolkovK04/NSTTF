#include "gtest/gtest.h"

#include <CL/cl.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>

#include "../src/tensor/tensor.h"
#include "../src/utils/functions.h"

#include <vector>

using namespace NSTTF;
class OperationTests : public ::testing::Test {
protected:
gpu::Context context;
  virtual void SetUp() {
    // Initialize OpenCL context, command queue, and other resources
    // This code is specific to your OpenCL setup and platform

    std::vector<gpu::Device> devices = gpu::enumDevices();

    gpu::Device device = devices[devices.size() - 1];


    context.init(device.device_id_opencl);
    context.activate();
  }
};

TEST_F(OperationTests, PositiveSumTest) {
  Tensor a(std::vector<float>{1, 2, 3});
  Tensor b(std::vector<float>{4, 5, 6});
  Tensor c = sum({a, b});
  gpu::gpu_mem_32f buff = c.getGPUBuffer();
  std::vector<float> v(3);
  buff.readN(v.data(), 3);
  EXPECT_EQ(v[0], 5);
  EXPECT_EQ(v[1], 7);
  EXPECT_EQ(v[2], 9);
}

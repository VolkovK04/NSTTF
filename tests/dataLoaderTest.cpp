#include "gtest/gtest.h"

#include <CL/cl.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>

#include <operations/function.h>
#include <tensor/tensor.h>

#include <dataLoader/dataLoader.h>
#include <vector>

using namespace NSTTF;

using namespace NSTTF;
class DataLoaderTest : public ::testing::Test {
protected:
  gpu::Context context;
  virtual void SetUp() {
    // Initialize OpenCL context, command queue, and other resources
    // This code is specific to your OpenCL setup and platform

    std::vector<gpu::Device> devices = gpu::enumDevices();

    gpu::Device device = devices[devices.size() - 1];

    context.init(device.device_id_opencl);
    context.activate();

    init();
  }
};

TEST_F(DataLoaderTest, baseSizeCheck) {
  // Load MNIST data;
  MNIST_DataLoader train_data_loader("train");
  MNIST_DataLoader test_data_loader("test");

  EXPECT_EQ(47040000, train_data_loader.dataset->images.size());
  EXPECT_EQ(60000, train_data_loader.dataset->labels.size());
  EXPECT_EQ(7840000, test_data_loader.dataset->images.size());
  EXPECT_EQ(10000, test_data_loader.dataset->labels.size());
}

TEST_F(DataLoaderTest, operator) {
  // Load MNIST data;
  MNIST_DataLoader train_data_loader("train");
  MNIST_DataLoader test_data_loader("test");

  EXPECT_THROW(train_data_loader[60000], std::out_of_range);
  EXPECT_THROW(test_data_loader[10000], std::out_of_range);

  try {
    train_data_loader[59999];
  } catch (...) {
    FAIL();
  }
}

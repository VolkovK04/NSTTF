#include "gtest/gtest.h"

#include <CL/cl.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>

#include <tensor/tensor.h>
#include <libgpu/shared_device_buffer.h>
#include <operations/function.h>
#include <vector>

using namespace NSTTF;
class TensorTests : public ::testing::Test {
protected:
  gpu::Context context;
  virtual void SetUp_() {
    // Initialize OpenCL context, command queue, and other resources
    // This code is specific to your OpenCL setup and platform

    std::vector<gpu::Device> devices = gpu::enumDevices();

    gpu::Device device = devices[devices.size() - 1];

    context.init(device.device_id_opencl);
    context.activate();

    NSTTF::init();
  }
};

TEST_F(TensorTests, universalCtorTest) {
  Tensor tensor{{1.f, 2.f, 3.f}, {1, 3}};
  std::vector<size_t> expectedShape{1, 3};
  std::vector<float> expectedData{1.f, 2.f, 3.f};
  EXPECT_EQ(tensor.getData(), expectedData);
  EXPECT_EQ(tensor.getShape(), expectedShape);
}

TEST_F(TensorTests, shapeCtorTest) {
  std::vector<size_t> expected{1, 2, 3};
  Tensor tensor{expected};
  EXPECT_EQ(tensor.getShape(), expected);
}

TEST_F(TensorTests, floatVectorCtorTest) {
  std::vector<float> expectedValues{1.f, 2.f, 3.f};
  Tensor tensor(expectedValues);
  std::vector<size_t> expectedShape(1, 3);
  EXPECT_EQ(tensor.getShape(), expectedShape);
  EXPECT_EQ(tensor.getData(), expectedValues);
}

TEST_F(TensorTests, incompatibleShapes) {
  try {
    Tensor tensor{{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}, {3, 2}};
    FAIL() << "Expected std::invalid_argument";
  } catch (std::invalid_argument &err) {
    EXPECT_EQ(err.what(), std::string("Invalid data"));
  } catch (...) {
    FAIL() << "Expected std::invalid_argument";
  }
}

TEST_F(TensorTests, invalidShape) {
  std::vector<size_t> init;
  try {
    Tensor tensor{init};
    FAIL() << "No exception";
  } catch (std::invalid_argument &err) {
    EXPECT_EQ(err.what(), std::string("Invalid shape"));
  } catch (...) {
    FAIL() << "Expected std::invalid_argument ahaaha";
  }
}

TEST_F(TensorTests, possibleReshape) {
  Tensor tensor{{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}, {1, 9}};
  tensor.reshape({3, 3});
  std::vector<size_t> expectedShape{3, 3};
  EXPECT_EQ(expectedShape, tensor.getShape());
}

TEST_F(TensorTests, impossibleReshape) {
  Tensor tensor{{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}, {1, 9}};
  try {
    tensor.reshape({3, 2});
    FAIL() << "No exception";
  } catch (std::invalid_argument &err) {
    EXPECT_EQ(err.what(), std::string("Incompatible shapes"));
  } catch (...) {
    FAIL() << "Another exception";
  }
}
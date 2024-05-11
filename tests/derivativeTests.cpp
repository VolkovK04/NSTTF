#include "gtest/gtest.h"

#include <CL/cl.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>

#include <compiler/compiler.h>
#include <executor/graphExecutor.h>
#include <operations/function.h>
#include <tensor/tensor.h>

using namespace NSTTF;

class DerivativeTests : public ::testing::Test {
protected:
  gpu::Context context;
  virtual void SetUp() {
    // Initialize OpenCL context, command queue, and other resources
    // This code is specific to your OpenCL setup and platform

    std::vector<gpu::Device> devices = gpu::enumDevices();

    gpu::Device device = devices[devices.size() - 1];

    context.init(device.device_id_opencl);
    context.activate();
    NSTTF::init();
  }
};

TEST_F(DerivativeTests, sumTest) {
  Compiler compiler;
  ComputationGraph g;
  Tensor test1({1.f}, {1});
  Tensor test2({4.f}, {1});
  TensorMap tensorsMap = {{"test1", test1}, {"test2", test2}};

  NodeInterface nodeInterface1 = g.AddInputNode("test1");
  NodeInterface nodeInterface2 = g.AddInputNode("test2");
  NodeInterface sumNode = nodeInterface1 + nodeInterface2;
  sumNode.setOutput();
  sumNode.setName("loss");

  GraphExecutorWG gewg = compiler.compileWithGrads(g);
  TensorMap actualForward = gewg.execute(tensorsMap);

  TensorMap actualDerivative = gewg.executeGrads();
  std::vector test1_data = actualDerivative.at("~grad_test1").getData();
  std::vector test2_data = actualDerivative.at("~grad_test2").getData();
}

TEST_F(DerivativeTests, hellTest) {
  Compiler compiler;
  ComputationGraph g;
  Tensor test1({1.f}, {1});
  Tensor test2({4.f}, {1});
  Tensor test3({3.f}, {1});
  Tensor test4({2.f}, {1});
  TensorMap tensorsMap = {
      {"test1", test1}, {"test2", test2}, {"test3", test3}, {"test4", test4}};

  NodeInterface nodeInterface1 = g.AddInputNode("test1");
  NodeInterface nodeInterface2 = g.AddInputNode("test2");
  NodeInterface nodeInterface3 = g.AddInputNode("test3");
  NodeInterface nodeInterface4 = g.AddInputNode("test4");

  NodeInterface a = nodeInterface1 + nodeInterface2;
  a.setName("a");

  NodeInterface b = nodeInterface3 - nodeInterface4;
  b.setName("b");

  NodeInterface c = a * b;
  c.setName("c");

  // NodeInterface fake = a * a;
  // fake.setOutput();

  NodeInterface d = a + c;
  d.setOutput();
  d.setName("loss");



  GraphExecutorWG gewg = compiler.compileWithGrads(g);
  TensorMap actualForward = gewg.execute(tensorsMap);

  TensorMap actualDerivative = gewg.executeGrads();
  std::vector test1_data = actualDerivative.at("~grad_test1").getData(); // 2
  std::vector test2_data = actualDerivative.at("~grad_test2").getData(); // 2
  std::vector test3_data = actualDerivative.at("~grad_test3").getData(); // 5
  std::vector test4_data = actualDerivative.at("~grad_test4").getData(); // -5
}
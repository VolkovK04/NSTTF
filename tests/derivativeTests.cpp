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
  ComputationGraph g;
  NodeInterface nodeInterface1 = g.AddInputNode("test1");
  NodeInterface nodeInterface2 = g.AddInputNode("test2");
  NodeInterface sumNode = nodeInterface1 + nodeInterface2;
  sumNode.setOutput();
  sumNode.setName("loss");

  Compiler compiler;
  GraphExecutorWG gewg = compiler.compileWithGrads(g);

  TensorMap tensorsMap = {{"test1", 1.f}, {"test2", 4.f}};
  TensorMap actualForward = gewg.execute(tensorsMap);
  EXPECT_EQ(actualForward.at("loss").getData(), std::vector<float>{5.f});

  TensorMap actualDerivative = gewg.executeGrads();
  EXPECT_EQ(actualDerivative.at("~grad_test1").getData(),
            std::vector<float>{1.f});
  EXPECT_EQ(actualDerivative.at("~grad_test2").getData(),
            std::vector<float>{1.f});
}

TEST_F(DerivativeTests, UltimateDerivativeTest) {
  // stage 1: create computation graph
  ComputationGraph g;

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

  NodeInterface fake = a * a;
  fake.setOutput();
  fake.setName("fake_output");

  NodeInterface d = a + c;
  d.setOutput();
  d.setName("loss");

  // TODO add check for graph

  // stage 2: compile existing graph
  Compiler compiler;

  GraphExecutorWG gewg = compiler.compileWithGrads(g);

  // gewg.printInstructions(std::cout); // for debug (unspecified)
  // gewg.printGradInstructions(std::cout); // for debug (unspecified)

  // stage 3: execute compiled grahp executor

  // test1
  TensorMap tensorsMap = {
      {"test1", -2.f}, {"test2", -0.5f}, {"test3", 4.f}, {"test4", -5.f}};

  TensorMap actualForward = gewg.execute(tensorsMap);

  EXPECT_EQ(actualForward.at("loss").getData(),
            std::vector<float>{-25.f}); // (T1 + T2)(T3 - T4 + 1)
  EXPECT_EQ(actualForward.at("fake_output").getData(),
            std::vector<float>{6.25f}); // (T1 + T2)^2

  // backward
  TensorMap actualDerivative = gewg.executeGrads();

  EXPECT_EQ(actualDerivative.at("~grad_test1").getData(),
            std::vector<float>{10.f}); // (T3 - T4 + 1)
  EXPECT_EQ(actualDerivative.at("~grad_test2").getData(),
            std::vector<float>{10.f}); // (T3 - T4 + 1)
  EXPECT_EQ(actualDerivative.at("~grad_test3").getData(),
            std::vector<float>{-2.5f}); // T1 + T2
  EXPECT_EQ(actualDerivative.at("~grad_test4").getData(),
            std::vector<float>{2.5f}); // - T1 - T2

  // test2
  // forward
  tensorsMap = {{"test1", 1.f}, {"test2", 4.f}, {"test3", 3.f}, {"test4", 2.f}};

  actualForward = gewg.execute(tensorsMap);

  EXPECT_EQ(actualForward.at("loss").getData(),
            std::vector<float>{10.f}); // (T1 + T2)(T3 - T4 + 1)
  EXPECT_EQ(actualForward.at("fake_output").getData(),
            std::vector<float>{25.f}); // (T1 + T2)^2

  // backward
  actualDerivative = gewg.executeGrads();

  EXPECT_EQ(actualDerivative.at("~grad_test1").getData(),
            std::vector<float>{2.f}); // (T3 - T4 + 1)
  EXPECT_EQ(actualDerivative.at("~grad_test2").getData(),
            std::vector<float>{2.f}); // (T3 - T4 + 1)
  EXPECT_EQ(actualDerivative.at("~grad_test3").getData(),
            std::vector<float>{5.f}); // T1 + T2
  EXPECT_EQ(actualDerivative.at("~grad_test4").getData(),
            std::vector<float>{-5.f}); // - T1 - T2
}

TEST_F(DerivativeTests, reduceSumDerivative) {
  ComputationGraph g;
  NodeInterface nodeInterface = g.AddInputNode("test1");
  NodeInterface rs = NodeInterface::ReduceSum(nodeInterface);
  rs.setName("loss");
  rs.setOutput();

  Compiler compiler;
  GraphExecutorWG gewg = compiler.compileWithGrads(g);

  std::vector<float> inputData{1.f, 2.f, 3.f, 4.f};
  std::vector<size_t> shape{2, 2, 1};

  Tensor input(inputData, shape);

  TensorMap tensorsMap;
  tensorsMap["test1"] = input;
  TensorMap actualForward = gewg.execute(tensorsMap);
  std::vector<float> res{4, 6};
  EXPECT_EQ(actualForward.at("loss").getData(), res);

  TensorMap actualDerivative = gewg.executeGrads();

  std::vector<float> expected{
      1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
  };

  EXPECT_EQ(actualDerivative["~grad_test1"].getData(), expected);
}
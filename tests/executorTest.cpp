#include "gtest/gtest.h"

#include <CL/cl.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>

#include "../src/computationGraph/computationGraph.h"
#include "../src/utils/compiler.h"
#include "../src/utils/graphExecutor.h"

#include <map>

using namespace NSTTF;

class ExecutorTests : public ::testing::Test {
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

TEST_F(ExecutorTests, SumNode) {
  Compiler compiler;
  ComputationGraph g;
  std::map<std::string, Tensor> tensorsMap = {
      {"test1", Tensor{{1.f, 2.f, 3.f}, {1, 3}}},
      {"test2", Tensor{{4.f, 5.f, 6.f}, {1, 3}}}};

  NodeInterface nodeInterface1 = g.AddInputNode("test1");
  NodeInterface nodeInterface2 = g.AddInputNode("test2");
  NodeInterface sumNode = nodeInterface1 + nodeInterface2;
  sumNode.setOutput();
  sumNode.setName("result");

  GraphExecutor executor = compiler.compile(g);
  std::map<std::string, Tensor> actual = executor.execute(tensorsMap);

  std::map<std::string, Tensor> expected = {
      {"test1", Tensor{{1.f, 2.f, 3.f}, {1, 3}}},
      {"test2", Tensor{{4.f, 5.f, 6.f}, {1, 3}}},
      {"result", Tensor{{5.f, 7.f, 9.f}, {1, 3}}}};

  Tensor actTest1 = actual["test1"];
  Tensor actTest2 = actual["test2"];
  Tensor actRes = actual["result"];

  Tensor expTest1 = expected["test1"];
  Tensor expTest2 = expected["test2"];
  Tensor expRes = expected["result"];

  EXPECT_EQ(actTest1.getData(), expTest1.getData());
  EXPECT_EQ(actTest2.getData(), expTest2.getData());
  EXPECT_EQ(actRes.getData(), expRes.getData());
}
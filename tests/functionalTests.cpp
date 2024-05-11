#include "gtest/gtest.h"

#include <CL/cl.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>

#include <compiler/compiler.h>
#include <computationGraph/computationGraph.h>
#include <executor/graphExecutor.h>
#include <operations/function.h>
#include <tensor/tensor.h>

#include <vector>

using namespace NSTTF;

// Test case 1:
//  Task: Try to do some operations without init GPU context.
//  Expected: Error("No GPU context!")

TEST(FunctionalTest, operationWithoutInit) {
  try {
    Tensor a(std::vector<float>{2, 3});
    FAIL() << "Dont get exception";
  } catch (std::runtime_error &err) {
    std::cout << err.what() << std::endl;
  } catch (...) {
    FAIL() << "Expected std::runtime_error";
  }
}

// Test case 2:
//  Task:
//    1: Init device
//    2: Init two tensors by value and size
//    3: Substract them, result to new Tensor
//    4: Retrieve and read GPU buffer to vector
//    5: Get correct answer
//  Expected: Correct work

TEST(FunctionalTest, simpleSubstruct) {
  // task 1:
  gpu::Context context;

  std::vector<gpu::Device> devices = gpu::enumDevices();

  gpu::Device device = devices[devices.size() - 1];

  context.init(device.device_id_opencl);
  context.activate();
  NSTTF::init();

  // task 2:
  std::vector<size_t> shape = {3, 2};
  Tensor a(std::vector<float>{1, 2, 6, 5, 6, 8}, shape);
  Tensor b(std::vector<float>{4, 5, 1, -9, 0, 10}, shape);

  // task 3:
  Tensor c = functions.at("subtraction")->compute({a, b});

  // task 4:
  gpu::gpu_mem_32f buff = c.getGPUBuffer();
  std::vector<float> v(c.getSize());
  buff.readN(v.data(), c.getSize());

  // task 5:
  EXPECT_EQ(c.getShape(), shape);

  EXPECT_EQ(v[0], -3);
  EXPECT_EQ(v[1], -3);
  EXPECT_EQ(v[2], 5);
  EXPECT_EQ(v[3], 14);
  EXPECT_EQ(v[4], 6);
  EXPECT_EQ(v[5], -2);
}

// Test case 3:
//  Task:
//    1: Init device
//    2: Init TensorMap with two tensors
//    3: Init two input nodes
//    4: Init multiplyNode from two input nodes, set it to output
//    5: Call ̷S̶̷̶a̶̷̶u̶̷̶l compiler end execute TensorMap
//    6: Get correct answer
//  Expected: Correct work

TEST(FunctionalTest, multiplyNodes) {
  // task1:
  gpu::Context context;

  std::vector<gpu::Device> devices = gpu::enumDevices();

  gpu::Device device = devices[devices.size() - 1];

  context.init(device.device_id_opencl);
  context.activate();
  NSTTF::init();

  // task2:
  Compiler compiler;
  ComputationGraph g;
  TensorMap tensorsMap = {{"first", Tensor{{1.f, 2.f, 3.f}, {1, 3}}},
                          {"second", Tensor{{4.f, 0.f, -1.f}, {1, 3}}}};

  // task3:
  NodeInterface nodeInterface1 = g.AddInputNode("first");
  NodeInterface nodeInterface2 = g.AddInputNode("second");

  // task4:
  NodeInterface multNode = nodeInterface1 * nodeInterface2;
  multNode.setOutput();
  multNode.setName("result");

  // task5:
  GraphExecutor executor = compiler.compile(g);
  TensorMap actual = executor.execute(tensorsMap);

  // task6:
  TensorMap expected = {{"result", Tensor{{4.f, 0.f, -3.f}, {1, 3}}}};

  Tensor actRes = actual["result"];

  Tensor expRes = expected["result"];

  EXPECT_EQ(actRes.getData(), expRes.getData());
}
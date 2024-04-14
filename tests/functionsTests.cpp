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

TEST_F(OperationTests, sumEmptyTensors) {
    try {
        sum({});
        FAIL() << "Dont get exception";
    } catch (std::runtime_error &err) {
        EXPECT_EQ(err.what(), std::string("Not enought tensors"));
    } catch (...) {
        FAIL() << "Expected std::runtime_error";
    }
}

TEST_F(OperationTests, sumWrongSize1) {
    Tensor a(std::vector<float>{2, 3});
    Tensor b(std::vector<float>{4, 5, 6});

    try {
        sum({a, b});
        FAIL() << "Dont get exception";
    } catch (std::runtime_error &err) {
        EXPECT_EQ(err.what(), std::string("Different shape"));
    } catch (...) {
        FAIL() << "Expected std::runtime_error";
    }
}

TEST_F(OperationTests, sumWrongSize2) {
    Tensor a(std::vector<float>{2, 3});
    Tensor b(std::vector<float>{4});

    try {
        sum({a, b});
        FAIL() << "Dont get exception";
    } catch (std::runtime_error &err) {
        EXPECT_EQ(err.what(), std::string("Different shape"));
    } catch (...) {
        FAIL() << "Expected std::runtime_error";
    }
}

TEST_F(OperationTests, sumWrongSize3) {
    Tensor a(std::vector<float>{2, 3});

    try {
        sum({a});
        FAIL() << "Dont get exception";
    } catch (std::runtime_error &err) {
        EXPECT_EQ(err.what(), std::string("Not enought tensors"));
    } catch (...) {
        FAIL() << "Expected std::runtime_error";
    }
}

TEST_F(OperationTests, sumWrongSize4) {
    Tensor a(std::vector<float>{2, 3});
    Tensor b(std::vector<float>{2, 3});
    Tensor c(std::vector<float>{2, 3});

    try {
        sum({a, b, c});
        FAIL() << "Dont get exception";
    } catch (std::runtime_error &err) {
        EXPECT_EQ(err.what(), std::string("Too many tensors"));
    } catch (...) {
        FAIL() << "Expected std::runtime_error";
    }
}

TEST_F(OperationTests, sumPositiveTest) {
    Tensor a(std::vector<float>{1, 2, 3});
    Tensor b(std::vector<float>{4, 5, 6});
    Tensor c = sum({a, b});

    gpu::gpu_mem_32f buff = c.getGPUBuffer();
    std::vector<float> v(c.getSize());
    buff.readN(v.data(), c.getSize());

    EXPECT_EQ(v[0], 5);
    EXPECT_EQ(v[1], 7);
    EXPECT_EQ(v[2], 9);
}


TEST_F(OperationTests, subtractionWrongSize1) {
    Tensor a(std::vector<float>{2, 3});
    Tensor b(std::vector<float>{2, 3, 4});

    try {
        subtraction({a, b});
        FAIL() << "Dont get exception";
    } catch (std::runtime_error &err) {
        EXPECT_EQ(err.what(), std::string("Different shape"));
    } catch (...) {
        FAIL() << "Expected std::runtime_error";
    }
}

TEST_F(OperationTests, subtractionWrongSize2) {
    Tensor a(std::vector<float>{2, 3});
    Tensor b(std::vector<float>{2, 3});
    Tensor c(std::vector<float>{2, 3});

    try {
        subtraction({a, b, c});
        FAIL() << "Dont get exception";
    } catch (std::runtime_error &err) {
        EXPECT_EQ(err.what(), std::string("Too many tensors"));
    } catch (...) {
        FAIL() << "Expected std::runtime_error";
    }
}

TEST_F(OperationTests, subtractionPositiveTest) {
    Tensor a(std::vector<float>{1, 2, 6});
    Tensor b(std::vector<float>{4, 5, 1});
    Tensor c = subtraction({a, b});

    gpu::gpu_mem_32f buff = c.getGPUBuffer();
    std::vector<float> v(c.getSize());
    buff.readN(v.data(), c.getSize());

    EXPECT_EQ(v[0], -3);
    EXPECT_EQ(v[1], -3);
    EXPECT_EQ(v[2], 5);
}


TEST_F(OperationTests, multiplicationWrongSize1) {
    Tensor a({2, 3, 3, 4}, {2, 2});
    Tensor b({2, 3, 4}, {1, 3});

    try {
        multiplication({a, b});
        FAIL() << "Dont get exception";
    } catch (std::runtime_error &err) {
        EXPECT_EQ(err.what(), std::string("Different shape"));
    } catch (...) {
        FAIL() << "Expected std::runtime_error";
    }
}

TEST_F(OperationTests, multiplicationWrongSize2) {
    Tensor a(std::vector<float>{2, 3});

    try {
        multiplication({a});
        FAIL() << "Dont get exception";
    } catch (std::runtime_error &err) {
        EXPECT_EQ(err.what(), std::string("Not enought tensors"));
    } catch (...) {
        FAIL() << "Expected std::runtime_error";
    }
}

TEST_F(OperationTests, multiplicationPositiveTest) {
    Tensor a(std::vector<float>{1, 2, 6});
    Tensor b(std::vector<float>{4, 5, 4});
    Tensor c = multiplication({a, b});

    gpu::gpu_mem_32f buff = c.getGPUBuffer();
    std::vector<float> v(c.getSize());
    buff.readN(v.data(), c.getSize());

    EXPECT_EQ(v[0], 4);
    EXPECT_EQ(v[1], 10);
    EXPECT_EQ(v[2], 24);
}


TEST_F(OperationTests, matrix_multiplicationWrongSize1) {
    Tensor a({2, 3}, {1, 2});
    Tensor b({2, 3, 4}, {1, 3});

    try {
        matrix_multiplication({a, b});
        FAIL() << "Dont get exception";
    } catch (std::runtime_error &err) {
        EXPECT_EQ(err.what(), std::string("Wrong matrix shape"));
    } catch (...) {
        FAIL() << "Expected std::runtime_error";
    }
}

TEST_F(OperationTests, matrix_multiplicationWrongSize2) {
    Tensor a(std::vector<float>{2, 3});

    try {
        matrix_multiplication({a});
        FAIL() << "Dont get exception";
    } catch (std::runtime_error &err) {
        EXPECT_EQ(err.what(), std::string("Not enought tensors"));
    } catch (...) {
        FAIL() << "Expected std::runtime_error";
    }
}

TEST_F(OperationTests, matrix_multiplicationWrongShape) {
    Tensor a({5, 6}, {1, 2});
    Tensor b({7, 8}, {1, 2});

    try {
        matrix_multiplication({a, b});
        FAIL() << "Dont get exception";
    } catch (std::runtime_error &err) {
        EXPECT_EQ(err.what(), std::string("Wrong matrix shape"));
    } catch (...) {
        FAIL() << "Expected std::runtime_error";
    }
}

TEST_F(OperationTests, matrix_multiplicationPositiveTest1) {
    Tensor a({2}, {1, 1});
    Tensor b({4}, {1, 1});
    Tensor c = matrix_multiplication({a, b});

    gpu::gpu_mem_32f buff = c.getGPUBuffer();
    std::vector<float> v(c.getSize());
    buff.readN(v.data(), c.getSize());

    EXPECT_EQ(v[0], 8);
}

TEST_F(OperationTests, matrix_multiplicationPositiveTest2) {
    Tensor a({1, 4}, {1, 2});
    Tensor b({5, 1}, {2, 1});
    Tensor c = matrix_multiplication({a, b});
    std::vector<size_t> expected{1, 1};

    gpu::gpu_mem_32f buff = c.getGPUBuffer();
    std::vector<float> v(c.getSize());
    buff.readN(v.data(), c.getSize());

    EXPECT_EQ(c.getShape(), expected);
    EXPECT_EQ(v[0], 9);
}

TEST_F(OperationTests, matrix_transposePositiveTest1) {
    Tensor a({2, 3, 4}, {1, 3});
    Tensor c = matrix_transpose({a});
    std::vector<size_t> expected{3, 1};

    gpu::gpu_mem_32f buff = c.getGPUBuffer();
    std::vector<float> v(c.getSize());
    buff.readN(v.data(), c.getSize());

    EXPECT_EQ(c.getShape(), expected);
    EXPECT_EQ(v[0], 2);
    EXPECT_EQ(v[1], 3);
    EXPECT_EQ(v[2], 4);
}


TEST_F(OperationTests, matrix_transposePositiveTest2) {
    Tensor a({2, 3, 4, 5}, {2, 2});
    Tensor c = matrix_transpose({a});
    std::vector<size_t> expected{2, 2};

    gpu::gpu_mem_32f buff = c.getGPUBuffer();
    std::vector<float> v(c.getSize());
    buff.readN(v.data(), c.getSize());

    EXPECT_EQ(c.getShape(), expected);
    EXPECT_EQ(v[0], 2);
    EXPECT_EQ(v[1], 4);
    EXPECT_EQ(v[2], 3);
    EXPECT_EQ(v[3], 5);
}
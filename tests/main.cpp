#include <CL/cl.h>
#include "gtest/gtest.h"

#include "../src/cl_build_headers/sum_cl.h"
#include "../src/cl_build_headers/matrix_transpose_cl.h"
#include "../src/cl_build_headers/subtraction_cl.h"
#include "../src/cl_build_headers/multiplication_cl.h"
#include "../src/cl_build_headers/matrix_multiplication_cl.h"


class OpenCLTestFixture : public ::testing::Test {
protected:
    cl_context context;
    cl_command_queue queue;
    // ... other OpenCL setup variables

    virtual void SetUp() {
        // Initialize OpenCL context, command queue, and other resources
        // This code is specific to your OpenCL setup and platform
    }

    virtual void TearDown() {
        // Release OpenCL resources
        // Again, this is specific to your setup
    }
};

TEST_F(OpenCLTestFixture, KernelExecutionTest) {
    // Load and compile the kernel from your src/cl directory
    // Create buffers, set arguments, etc.

    // Execute the kernel and wait for it to finish
    // clEnqueueNDRangeKernel(...);
    // clFinish(queue);

    // Read back the results and ASSERT or EXPECT certain conditions
    // This is where you check if the kernel's output is as expected
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
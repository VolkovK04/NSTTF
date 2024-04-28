#include "gtest/gtest.h"

#include <CL/cl.h>

#include <chrono>
#include <cmath>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>

#include <tensor/tensor.h>
#include <operations/function.h>

#include <vector>

using namespace NSTTF;

class EfficiencyTests : public ::testing::Test {
  protected:
    gpu::Context context;
    //   virtual void SetUp() {
    //     // Initialize OpenCL context, command queue, and other resources
    //     // This code is specific to your OpenCL setup and platform

    //     // std::vector<gpu::Device> devices = gpu::enumDevices();

    //     // gpu::Device device = devices[devices.size() - 1];

    //     // context.init(device.device_id_opencl);
    //     // context.activate();

    //
    //   }

    void setUpDevice(int i) {
        std::vector<gpu::Device> devices = gpu::enumDevices();

        gpu::Device device = devices[i];

        context.init(device.device_id_opencl);
        context.activate();

        device.printInfo();
    }
};

template <class DT = std::chrono::milliseconds,
          class ClockT = std::chrono::steady_clock>
class Timer {
    using timep_t = decltype(ClockT::now());

    timep_t _start = ClockT::now();
    timep_t _end = {};

  public:
    void start() {
        _end = timep_t{};
        _start = ClockT::now();
    }

    void end() { _end = ClockT::now(); }

    template <class duration_t = DT> auto duration() const {
        // Use gsl_Expects if your project supports it.
        assert(_end != timep_t{} && "Timer must toc before reading the time");
        return std::chrono::duration_cast<duration_t>(_end - _start);
    }
};

TEST_F(EfficiencyTests, substruction) {
    std::cout << "CTEST_FULL_OUTPUT" << std::endl;

    int numOfDevices = gpu::enumDevices().size();
    int vectorSize = 100000000;
    int benchIters = 1;
    Timer clock;

    std::vector<float> v1, v2, v3;

    FastRandom r(vectorSize * 2);
    for (unsigned int i = 0; i < vectorSize; ++i) {
        v1.push_back(r.nextf());
        v3.push_back(v1.back());
    }

    for (unsigned int i = 0; i < vectorSize; ++i) {
        v2.push_back(r.nextf());
        v3[i] += v2.back();
    }


    for (unsigned int i = 0; i < numOfDevices; ++i) {
        setUpDevice(i);
        init();

        Tensor a(v1);
        Tensor b(v2);
        Tensor c;

        clock.start();
        c = functions_.at("subtraction")->compute({a, b})[0];
        clock.end();

        std::cout << "Code run for " << clock.duration().count() << " ms"
                  << std::endl
                  << std::endl;

        

        gpu::gpu_mem_32f buff = c.getGPUBuffer();
        std::vector<float> result(c.getSize());
        buff.readN(result.data(), c.getSize());

        std::cout << "\n" << v3[0] << "  ----  " <<  result[0] << "\n";
        std::cout << "\n" << v3[1] << "  ----  " <<  result[1] << "\n";
        std::cout << "\n" << v3[2] << "  ----  " <<  result[2] << "\n";
        std::cout << "\n" << v3[3] << "  ----  " <<  result[3] << "\n";

        double diff_sum = 0;
        for (int i = 0; i < vectorSize; ++i) {
            double a = result[i];
            double b = v3[i];
            if (a != 0.0 || b != 0.0) {
                double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
                diff_sum += diff;
            }
        }

        double diff_avg = diff_sum / (vectorSize);
        // EXPECT_TRUE(diff_avg <= 0.01);
        if (diff_avg > 0.01) {
            FAIL() << "diff_avg = " << diff_avg;
        }
    }
}
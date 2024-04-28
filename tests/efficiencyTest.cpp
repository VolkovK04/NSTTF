#include "gtest/gtest.h"

#include <CL/cl.h>

#include <chrono>
#include <cmath>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>

#include <operations/function.h>
#include <tensor/tensor.h>

#include <vector>

using namespace NSTTF;

// 0 - from CPU
// 1 - from GPU
const int firstDevice = 1;

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

class EfficiencyTests : public ::testing::Test {
  protected:
    gpu::Context context;

    virtual void SetUp(){
        std::vector<gpu::Device> devices = gpu::enumDevices();

        gpu::Device device = devices[devices.size() - 1];

        context.init(device.device_id_opencl);
        context.activate();

        init();
    }

    virtual void SetUp(int i) {
        std::vector<gpu::Device> devices = gpu::enumDevices();

        gpu::Device device = devices[i];

        context.init(device.device_id_opencl);
        context.activate();

        device.printInfo();

        // init();
    }

    void testVectorFunction(std::shared_ptr<NSTTF::Function> func,
                            const char &operation, const int &benchIters = 1) {
        std::cout << "CTEST_FULL_OUTPUT" << std::endl;

        int numOfDevices = gpu::enumDevices().size();
        int vectorSize = 100000000;
        Timer clock;

        std::vector<float> v1, v2, v3;

        for (unsigned int iter = 0; iter < benchIters; ++iter) {
            FastRandom r(vectorSize * 2);
            for (unsigned int i = 0; i < vectorSize; ++i) {
                v1.push_back(r.nextf());
                v3.push_back(v1.back());
            }

            for (unsigned int i = 0; i < vectorSize; ++i) {
                v2.push_back(r.nextf());
                switch (operation) {
                case '+':
                    v3[i] += v2.back();
                    break;
                case '-':
                    v3[i] -= v2.back();
                    break;
                case '*':
                    v3[i] *= v2.back();
                    break;
                }
            }

            for (unsigned int i = firstDevice; i < numOfDevices; ++i) {
                SetUp(i);

                Tensor a(v1);
                Tensor b(v2);
                Tensor c;

                clock.start();
                c = func->compute({a, b})[0];
                clock.end();

                std::cout << "Code run for " << clock.duration().count()
                          << " ms" << std::endl
                          << std::endl;

                gpu::gpu_mem_32f buff = c.getGPUBuffer();
                std::vector<float> result(c.getSize());
                buff.readN(result.data(), c.getSize());

                double diff_sum = 0;
                for (int i = 0; i < vectorSize; ++i) {
                    double f = result[i];
                    double s = v3[i];
                    if (f != 0.0 || s != 0.0) {
                        double diff = fabs(f - s) / std::max(fabs(f), fabs(s));
                        diff_sum += diff;
                    }
                }

                double diff_avg = diff_sum / (vectorSize);
                // EXPECT_TRUE(diff_avg <= 0.01);
                if (diff_avg > 0.01) {
                    FAIL() << "diff_avg = " << diff_avg;
                }
            }
            v1.clear();
            v2.clear();
            v3.clear();
        }
    }
};

TEST_F(EfficiencyTests, sum) {
    // std::cout << "AGA" << std::endl;
    std::shared_ptr<NSTTF::Function> func = functions_.at("sum");
    testVectorFunction(func, '+', 3);
}
TEST_F(EfficiencyTests, subtraction) {
    testVectorFunction(functions_.at("subtraction"), '-', 3);
}
TEST_F(EfficiencyTests, multiplication) {
    testVectorFunction(functions_.at("multiplication"), '*', 3);
}

TEST_F(EfficiencyTests, matrix_multiplication) {
    std::cout << "CTEST_FULL_OUTPUT" << std::endl;

    int numOfDevices = gpu::enumDevices().size();
    unsigned int M = 2048;
    unsigned int K = 2048;
    unsigned int N = 2048;
    unsigned int benchIters = 1;
    Timer clock;

    std::vector<float> v1, v2, v3;

    v1.resize(M * K, 0);
    v2.resize(K * N, 0);
    v3.resize(M * N, 0);

    for (unsigned int iter = 0; iter < benchIters; ++iter) {
        FastRandom r(M + K + N);
        for (unsigned int i = 0; i < v1.size(); ++i) {
            v1[i] = r.nextf();
        }
        for (unsigned int i = 0; i < v2.size(); ++i) {
            v2[i] = r.nextf();
        }

        // Compute matrix product
        for (int j = 0; j < M; ++j) {
            for (int i = 0; i < N; ++i) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += v1[j * K + k] * v2[k * N + i];
                }
                v3[j * N + i] = sum;
            }
        }

        for (unsigned int i = firstDevice; i < numOfDevices; ++i) {
            SetUp(i);

            Tensor a(v1);
            Tensor b(v2);
            Tensor c;

            a.reshape({M, K});
            b.reshape({K, N});

            clock.start();
            c = functions_.at("matrix_multiplication")->compute({a, b})[0];
            clock.end();

            std::cout << "Code run for " << clock.duration().count() << " ms"
                      << std::endl
                      << std::endl;

            gpu::gpu_mem_32f buff = c.getGPUBuffer();
            std::vector<float> result(c.getSize());
            buff.readN(result.data(), c.getSize());

            double diff_sum = 0;
            for (int i = 0; i < M * N; ++i) {
                double f = result[i];
                double s = v3[i];
                if (f != 0.0 || s != 0.0) {
                    double diff = fabs(f - s) / std::max(fabs(f), fabs(s));
                    diff_sum += diff;
                }
            }

            double diff_avg = diff_sum / (M * N);
            // EXPECT_TRUE(diff_avg <= 0.01);
            if (diff_avg > 0.01) {
                FAIL() << "diff_avg = " << diff_avg;
            }
        }
        v1.clear();
        v2.clear();
        v3.clear();
    }
}

TEST_F(EfficiencyTests, matrix_transpose) {
    std::cout << "CTEST_FULL_OUTPUT" << std::endl;

    int numOfDevices = gpu::enumDevices().size();
    unsigned int M = 4096;
    unsigned int N = 4096;
    unsigned int benchIters = 1;
    Timer clock;

    std::vector<float> v1, v2, v3;

    v1.resize(M * N, 0);
    v2.resize(N * M, 0);

    for (unsigned int iter = 0; iter < benchIters; ++iter) {
        FastRandom r(M + N);
        for (unsigned int i = 0; i < v1.size(); ++i) {
            v1[i] = r.nextf();
        }

        for (unsigned int i = firstDevice; i < numOfDevices; ++i) {
            SetUp(i);

            Tensor a(v1);
            Tensor c;

            a.reshape({M, N});

            clock.start();
            c = functions_.at("matrix_transpose")->compute({a})[0];
            clock.end();

            std::cout << "Code run for " << clock.duration().count() << " ms"
                      << std::endl
                      << std::endl;

            gpu::gpu_mem_32f buff = c.getGPUBuffer();
            std::vector<float> result(c.getSize());
            buff.readN(result.data(), c.getSize());

            for (int j = 0; j < M; ++j) {
                for (int i = 0; i < N; ++i) {
                    float f = result[j * N + i];
                    float s = v1[i * M + j];
                    if (f != s) {
                        FAIL() << "f = " << f << ", s = " << s;
                    }
                }
            }
        }
        v1.clear();
        v2.clear();
        v3.clear();
    }
}
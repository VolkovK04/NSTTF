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

#include <cstdlib>
#include <numeric>
#include <random>

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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

class EfficiencyTests : public ::testing::Test {
protected:
  gpu::Context context;
  std::vector<gpu::Device> devices = gpu::enumDevices();
  bool flag = false;

  void SetUp_(int deviceId) {
    gpu::Device device = devices[deviceId];

    context.init(device.device_id_opencl);
    context.activate();

    device.printInfo();

    init();
  }

  void testVectorFunction(std::shared_ptr<NSTTF::Function> func, char operation,
                          unsigned int benchIters = 1) {
    std::cout << "CTEST_FULL_OUTPUT" << std::endl;

    int vectorSize = 100'000'000; // Gflops ?
    Timer clock;

    for (unsigned int iter = 0; iter < benchIters; ++iter) {
      std::vector<float> v1, v2, v3;
      int seed = 0;
      FastRandom r(seed);
      for (size_t i = 0; i < vectorSize; ++i) {
        float x1 = r.nextf();
        float x2 = r.nextf();

        v1.push_back(x1);
        v2.push_back(x2);

        switch (operation) {
        case '+':
          v3.push_back(x1 + x2);
          break;
        case '-':
          v3.push_back(x1 - x2);
          break;
        case '*':
          v3.push_back(x1 * x2);
          break;
        }
      }

      for (size_t deviceId = firstDevice; deviceId < devices.size();
           ++deviceId) {
        SetUp_(deviceId);

        Tensor a(v1);
        Tensor b(v2);

        clock.start();
        Tensor c = func->compute({a, b});
        clock.end();

        std::cout << "Code run for " << clock.duration().count() << " ms"
                  << std::endl
                  << std::endl;

        std::vector<float> result = c.getData();

        double diff_sum = 0;
        for (size_t i = 0; i < vectorSize; ++i) {
          double f = result[i];
          double s = v3[i];
          if (f != 0.0 || s != 0.0) {
            double diff = fabs(f - s) / std::max(fabs(f), fabs(s));
            diff_sum += diff;
          }
        }

        double diff_avg = diff_sum / (vectorSize);

        // EXPECT_LE(diff_avg, 0.01);
        if (diff_avg > 0.01) {
          FAIL() << "diff_avg = " << diff_avg;
        }
      }
    }
  }
};

TEST_F(EfficiencyTests, sum) {
  testVectorFunction(functions.at("sum"), '+', 3);
}
TEST_F(EfficiencyTests, subtraction) {
  testVectorFunction(functions.at("subtraction"), '-', 3);
}
TEST_F(EfficiencyTests, multiplication) {
  testVectorFunction(functions.at("multiplication"), '*', 3);
}

TEST_F(EfficiencyTests, matrix_multiplication) {
  std::cout << "CTEST_FULL_OUTPUT" << std::endl;

  unsigned int M = 1024;
  unsigned int K = 1024;
  unsigned int N = 1024;
  unsigned int benchIters = 1;
  Timer clock;

  for (unsigned int iter = 0; iter < benchIters; ++iter) {
    std::vector<float> v1(M * K), v2(K * N), v3(M * N);

    FastRandom r;
    for (size_t i = 0; i < M * K; ++i) {
      v1[i] = r.nextf();
    }
    for (size_t i = 0; i < K * N; ++i) {
      v2[i] = r.nextf();
    }

    // Compute matrix product
    for (size_t j = 0; j < M; ++j) {
      for (size_t i = 0; i < N; ++i) {
        float sum = 0;
        for (size_t k = 0; k < K; ++k) {
          sum += v1[j * K + k] * v2[k * N + i];
        }
        v3[j * N + i] = sum;
      }
    }

    for (size_t deviceId = firstDevice; deviceId < devices.size(); ++deviceId) {
      SetUp_(deviceId);

      Tensor a(v1, {M, K});
      Tensor b(v2, {K, N});

      clock.start();
      Tensor c = functions.at("matrix_multiplication")->compute({a, b});
      clock.end();

      std::cout << "Code run for " << clock.duration().count() << " ms"
                << std::endl
                << std::endl;

      std::vector<float> result = c.getData();

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

      if (diff_avg > 0.01) {
        FAIL() << "diff_avg = " << diff_avg;
      }
    }
  }
}

TEST_F(EfficiencyTests, matrix_transpose) {
  std::cout << "CTEST_FULL_OUTPUT" << std::endl;

  unsigned int M = 4096;
  unsigned int N = 4096;
  unsigned int benchIters = 1;
  Timer clock;

  for (unsigned int iter = 0; iter < benchIters; ++iter) {
    std::vector<float> v1(M * N), v2(N * M);

    int seed = 0;
    FastRandom r(seed);
    for (size_t i = 0; i < v1.size(); ++i) {
      v1[i] = r.nextf();
    }

    for (size_t deviceId = firstDevice; deviceId < devices.size(); ++deviceId) {
      SetUp_(deviceId);

      Tensor a(v1, {M, N});

      clock.start();
      Tensor c = functions.at("matrix_transpose")->compute({a});
      clock.end();

      std::cout << "Code run for " << clock.duration().count() << " ms"
                << std::endl
                << std::endl;

      std::vector<float> result = c.getData();

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
  }
}

// Strange things happens with this method. Avoid using reduce_sum_1D for now
TEST_F(EfficiencyTests, LargeNumberOfElements1D) {
  SetUp_(devices.size() - 1);
  float minValue = 0.0f;
  float maxValue = 1.0f;

  std::random_device rd;
  std::mt19937 eng(rd());
  std::uniform_real_distribution<float> distr(minValue, maxValue);

  auto gen = [&]() { return distr(eng); };

  const size_t numElements = 100;
  std::vector<size_t> shape = {numElements};

  std::vector<float> v(numElements);
  std::generate(v.begin(), v.end(), gen);
  float resSum = std::accumulate(v.begin(), v.end(), 0.0f);

  Tensor expected(resSum);

  Tensor a(v, shape);
  Tensor res = functions.at("reduce_sum")->compute({a});
  std::vector<float> result = res.getData();

  EXPECT_EQ(res.getShape(), expected.getShape());
  EXPECT_NEAR(result[0], resSum, 1e-2);
}

TEST_F(EfficiencyTests, LargeNumberOfElements2D) {
  SetUp_(devices.size() - 1);
  float minValue = 0.0f;
  float maxValue = 1.0f;

  size_t minSize = 10;
  size_t maxSize = 100;

  std::random_device rd;
  std::mt19937 eng(rd());

  std::uniform_real_distribution<float> distrFloat(minValue, maxValue);
  std::uniform_int_distribution<size_t> distrSize(minSize, maxSize);

  auto genFloat = [&]() { return distrFloat(eng); };
  auto genSize = [&]() { return distrSize(eng); };

  std::vector<size_t> shape(3);
std:
  generate(shape.begin(), shape.end(), genSize);
  size_t size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

  std::vector<float> v(size);
  std::generate(v.begin(), v.end(), genFloat);

  Tensor a(v, shape);
  std::vector<size_t> resultedShape(shape.begin() + 1, shape.end());
  size_t resultedSize = size / shape[0];
  std::vector<float> expectedVector(resultedSize);

  for (size_t i = 0; i < resultedSize; ++i) {
    float sum = 0.0f;
    for (size_t j = 0; j < shape[0]; ++j) {
      sum += v[j * resultedSize + i];
    }
    expectedVector[i] = sum;
  }

  Tensor expectedTensor(expectedVector, resultedShape);
  Tensor res = functions.at("reduce_sum")->compute({a});
  std::vector<float> result = res.getData();

  EXPECT_EQ(res.getShape(), expectedTensor.getShape());

  EXPECT_EQ(result, expectedTensor.getData());
}

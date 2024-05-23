#include "gtest/gtest.h"
#include <CL/cl.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include <operations/function.h>

class GPUTests : public ::testing::Test {
protected:
  gpu::Context context;
  gpu::Device device;

  std::vector<float> as, bs, cs;
  gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;

  virtual void SetUp() {
    // Initialize OpenCL context, command queue, and other resources
    // This code is specific to your OpenCL setup and platform

    std::vector<gpu::Device> devices = gpu::enumDevices();
    device = devices[devices.size() - 1];

    context.init(device.device_id_opencl);
    context.activate();

    NSTTF::init();
  }

  virtual void TearDown() {
    // Release OpenCL resources
    // Again, this is specific to your setup
    context.clear();
    device = gpu::Device();

    if (!as.empty()) {
      as.clear();
    }
    if (!bs.empty()) {
      bs.clear();
    }
    if (!cs.empty()) {
      cs.clear();
    }
  }
};

TEST_F(GPUTests, multiplication_test) {
  unsigned int n = 1000;
  as.resize(n);
  bs.resize(n);
  cs.resize(n);

  FastRandom r(n);
  for (size_t i = 0; i < n; ++i) {
    as[i] = r.nextf();
    bs[i] = r.nextf();
  }

  as_gpu.resizeN(n);
  bs_gpu.resizeN(n);
  cs_gpu.resizeN(n);

  as_gpu.writeN(as.data(), n);
  bs_gpu.writeN(bs.data(), n);

  unsigned int workGroupSize = 128;
  unsigned int global_work_size =
      (n + workGroupSize - 1) / workGroupSize * workGroupSize;
  NSTTF::kernels.at("multiplication")
      .exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, bs_gpu,
            cs_gpu, n);

  cs_gpu.readN(cs.data(), n);

  // Проверяем корректность результатов
  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(cs[i], as[i] * bs[i]);
  }
}

TEST_F(GPUTests, sum_test) {
  unsigned int n = 1000;
  as.resize(n, 0);
  bs.resize(n, 0);
  cs.resize(n, 0);

  FastRandom r;
  for (size_t i = 0; i < n; ++i) {
    as[i] = r.nextf();
    bs[i] = r.nextf();
  }

  as_gpu.resizeN(n);
  bs_gpu.resizeN(n);
  cs_gpu.resizeN(n);

  as_gpu.writeN(as.data(), n);
  bs_gpu.writeN(bs.data(), n);

  unsigned int workGroupSize = 128;
  unsigned int global_work_size =
      (n + workGroupSize - 1) / workGroupSize * workGroupSize;
  NSTTF::kernels.at("sum").exec(gpu::WorkSize(workGroupSize, global_work_size),
                                as_gpu, bs_gpu, cs_gpu, n);

  cs_gpu.readN(cs.data(), n);

  // Проверяем корректность результатов
  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(cs[i], as[i] + bs[i]);
  }
}

TEST_F(GPUTests, subtraction_test) {
  unsigned int n = 1000;
  as.resize(n, 0);
  bs.resize(n, 0);
  cs.resize(n, 0);

  FastRandom r(n);
  for (size_t i = 0; i < n; ++i) {
    as[i] = r.nextf();
    bs[i] = r.nextf();
  }

  as_gpu.resizeN(n);
  bs_gpu.resizeN(n);
  cs_gpu.resizeN(n);

  as_gpu.writeN(as.data(), n);
  bs_gpu.writeN(bs.data(), n);

  unsigned int workGroupSize = 128;
  unsigned int global_work_size =
      (n + workGroupSize - 1) / workGroupSize * workGroupSize;
  NSTTF::kernels.at("subtraction")
      .exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, bs_gpu,
            cs_gpu, n);

  cs_gpu.readN(cs.data(), n);

  // Проверяем корректность результатов
  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(cs[i], as[i] - bs[i]);
  }
}

TEST_F(GPUTests, matrix_multiplication_updated_test) {
  unsigned int M = 128;
  unsigned int K = 128;
  unsigned int N = 128;

  as.resize(M * K, 0);
  bs.resize(K * N, 0);
  cs.resize(M * N, 0);

  FastRandom r;
  for (size_t i = 0; i < as.size(); ++i) {
    as[i] = r.nextf();
  }
  for (size_t i = 0; i < bs.size(); ++i) {
    bs[i] = r.nextf();
  }

  for (size_t j = 0; j < M; ++j) {
    for (size_t i = 0; i < N; ++i) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        sum += as.data()[j * K + k] * bs.data()[k * N + i];
      }
      cs.data()[j * N + i] = sum;
    }
  }

  const std::vector<float> cs_cpu_reference = cs;

  // Allocate memory on the GPU
  as_gpu.resizeN(M * K);
  bs_gpu.resizeN(K * N);
  cs_gpu.resizeN(M * N);

  as_gpu.writeN(as.data(), M * K);
  bs_gpu.writeN(bs.data(), K * N);

  unsigned int x_work_group_size = 8;
  unsigned int y_work_group_size = 8;
  unsigned int x_work_size =
      (M + x_work_group_size - 1) / x_work_group_size * x_work_group_size;
  unsigned int y_work_size =
      (N + y_work_group_size - 1) / y_work_group_size * y_work_group_size;

  NSTTF::kernels.at("matrix_multiplication")
      .exec(gpu::WorkSize(x_work_group_size, y_work_group_size, 1, x_work_size,
                          y_work_size, 1),
            as_gpu, bs_gpu, cs_gpu, 1, M, K, N);

  cs.resize(M * N);
  cs_gpu.readN(cs.data(), M * N);

  // Проверяем корректность результатов
  double diff_sum = 0;
  for (size_t i = 0; i < M * N; ++i) {
    double a = cs[i];
    double b = cs_cpu_reference[i];
    if (a != 0.0 || b != 0.0) {
      double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
      diff_sum += diff;
    }
  }

  double diff_avg = diff_sum / (M * N);

  if (diff_avg > 0.01) {
    FAIL() << "diff_avg = " << diff_avg;
  }
}

TEST_F(GPUTests, matrix_multiplication_full_test) {
  unsigned int L = 32;
  unsigned int M = 32;
  unsigned int K = 32;
  unsigned int N = 32;

  as.resize(L * M * K, 0);
  bs.resize(L * K * N, 0);
  cs.resize(L * M * N, 0);

  FastRandom r;
  for (size_t i = 0; i < as.size(); ++i) {
    as[i] = r.nextf();
  }
  for (size_t i = 0; i < bs.size(); ++i) {
    bs[i] = r.nextf();
  }

  for (size_t j = 0; j < M; ++j) {
    for (size_t i = 0; i < N; ++i) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        sum += as.data()[j * K + k] * bs.data()[k * N + i];
      }
      cs.data()[j * N + i] = sum;
    }
  }

  const std::vector<float> cs_cpu_reference = cs;

  // Allocate memory on the GPU
  as_gpu.resizeN(M * K);
  bs_gpu.resizeN(K * N);
  cs_gpu.resizeN(M * N);

  as_gpu.writeN(as.data(), M * K);
  bs_gpu.writeN(bs.data(), K * N);

  unsigned int x_work_group_size = 8;
  unsigned int y_work_group_size = 8;
  unsigned int x_work_size =
      (M + x_work_group_size - 1) / x_work_group_size * x_work_group_size;
  unsigned int y_work_size =
      (N + y_work_group_size - 1) / y_work_group_size * y_work_group_size;

  NSTTF::kernels.at("matrix_multiplication")
      .exec(gpu::WorkSize(x_work_group_size, y_work_group_size, 1, x_work_size,
                          y_work_size, 1),
            as_gpu, bs_gpu, cs_gpu, 1, M, K, N);

  cs.resize(M * N);
  cs_gpu.readN(cs.data(), M * N);

  // Проверяем корректность результатов
  double diff_sum = 0;
  for (size_t i = 0; i < M * N; ++i) {
    double a = cs[i];
    double b = cs_cpu_reference[i];
    if (a != 0.0 || b != 0.0) {
      double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
      diff_sum += diff;
    }
  }

  double diff_avg = diff_sum / (M * N);

  if (diff_avg > 0.01) {
    FAIL() << "diff_avg = " << diff_avg;
  }
}

TEST_F(GPUTests, matrix_transposition_test) {
  unsigned int M = 128;
  unsigned int K = 128;

  as.resize(M * K, 0);
  bs.resize(M * K, 0); // as = bs_t

  FastRandom r;
  for (size_t i = 0; i < as.size(); ++i) {
    as[i] = r.nextf();
  }

  as_gpu.resizeN(M * K);
  bs_gpu.resizeN(K * M);

  as_gpu.writeN(as.data(), M * K);

  unsigned int x_work_group_size = 16;
  unsigned int y_work_group_size = 16;
  unsigned int x_work_size =
      (M + x_work_group_size - 1) / x_work_group_size * x_work_group_size;
  unsigned int y_work_size =
      (K + y_work_group_size - 1) / y_work_group_size * y_work_group_size;

  NSTTF::kernels.at("matrix_transpose")
      .exec(gpu::WorkSize(x_work_group_size, y_work_group_size, x_work_size,
                          y_work_size),
            as_gpu, bs_gpu, M, K);

  bs_gpu.readN(bs.data(), M * K);

  for (size_t j = 0; j < M; ++j) {
    for (size_t i = 0; i < K; ++i) {
      float a = as[j * K + i];
      float b = bs[i * M + j];
      if (a != b) {
        FAIL() << "a = " << a << ", b = " << b;
      }
    }
  }
}
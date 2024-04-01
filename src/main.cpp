#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl_build_headers/sum_cl.h"
#include "cl_build_headers/matrix_multiplication_cl.h"
#include "cl_build_headers/matrix_transpose_cl.h"
#include "cl_build_headers/subtraction_cl.h"
#include "cl_build_headers/multiplication_cl.h"

#include <vector>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    unsigned int n = 50*1000*1000;
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;


    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);
    cs_gpu.resizeN(n);

    
    as_gpu.writeN(as.data(), n);
    bs_gpu.writeN(bs.data(), n);

    
    ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum");
    sum.compile();

    unsigned int workGroupSize = 128;
    unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
    sum.exec(gpu::WorkSize(workGroupSize, global_work_size),
                as_gpu, bs_gpu, cs_gpu, n);

    cs_gpu.readN(cs.data(), n);

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(cs[i], as[i] + bs[i], "GPU results should be equal to CPU results!");
    }


    return 0;
}

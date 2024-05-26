// #ifdef __CLION_IDE__
// #include <libgpu/opencl/cl/clion_defines.cl>
// #endif

__kernel void sum(__global const float *a, __global const float *b,
                  __global float *c, unsigned int n) {
  const unsigned int index = get_global_id(0);

  if (index >= n)
    return;

  c[index] = a[index] + b[index];
}
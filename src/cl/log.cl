__kernel void log_kernel(__global const float *a, __global float *result,
                         unsigned int n) {
  const unsigned int index = get_global_id(0);

  if (index >= n)
    return;

  result[index] = log(a[index]);
}
__kernel void log(__global const float *a, __global float *result, unsigned int n) {
  const unsigned int index = get_global_id(0);

  if (index >= n)
    return;

  result[index] = log(a[index]);
}
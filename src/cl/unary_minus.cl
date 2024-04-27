__kernel void unary_minus(__global const float *a, __global float *b,
                          unsigned int n) {
  const unsigned int index = get_global_id(0);

  if (index >= n)
    return;
  b[index] = -a[index];
}
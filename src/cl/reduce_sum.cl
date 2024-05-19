#define TILE_SIZE 128

//sum among z dimension
__kernel void reduce_sum_3D(__global const float *a, __global float *c,
                            unsigned int n) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  int iGlobal = i * n * n + j * n + k;
  float sum = 0.0f;
  for (int index = 0; index < n; index++) {
    sum += a[iGlobal];
    iGlobal += n * n;
  }
  c[i * n * n + j * n + k] = sum;
}


//sum among y dimension
__kernel void reduce_sum_2D(__global const float *a, __global float *c,
                            unsigned int n) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  int iGlobal = i * n + j;
  float sum = 0.0f;
  for (int index = 0; index < n; index++) {
    sum += a[iGlobal];
    iGlobal += n;
  }
  c[i * n + j] = sum;
}

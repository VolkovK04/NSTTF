#define TILE_SIZE 32

__kernel void reduce_sum_1D(__global const float *input,
                            __global float *partialSums,
                            const unsigned int numElements) {

  __local float sdata[TILE_SIZE];

  // Each thread loads one element from global to shared memory
  unsigned int tid = get_local_id(0);
  unsigned int i = get_global_id(0);

  // Load input into shared memory
  sdata[tid] = (i < numElements) ? input[i] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Do reduction in shared memory
  for (unsigned int s = TILE_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write result for this block to global memory
  if (tid == 0) {
    partialSums[get_group_id(0)] = sdata[0];
  }
}

__kernel void reduce_sum_2D(__global const float *in, __global float *out,
                            unsigned int axis_shape_size,
                            unsigned int resulted_shape_size) {

  size_t global_i = get_global_id(0);
  if (global_i >= resulted_shape_size) {
    return;
  }

  float sum = 0;
  for (size_t i = 0; i < axis_shape_size; ++i) {
    sum += in[i * resulted_shape_size + global_i];
  }
  out[global_i] = sum;
}

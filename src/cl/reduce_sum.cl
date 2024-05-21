#define CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT (1 << 17)

#define TILE_SIZE 32

__kernel void inner_sum(__global const float *in, __global float *partial_sums,
                        __local float *local_sums, unsigned int n) {
  size_t global_i = get_global_id(0);
  size_t local_i = get_local_id(0);

  if (global_i < n) {
    local_sums[local_i] = in[global_i];
  } else {
    local_sums[local_i] = 0.0f;
  }

  for (int i = TILE_SIZE / 2; i > 0; i /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_i < i) {
      local_sums[local_i] += local_sums[local_i + i];
    }
  }

  if (local_i == 0) {
    partial_sums[get_group_id(0)] = local_sums[0];
  }
}

__kernel void reduce_sum_1D(__global const float *in,
                            __global float *partial_sums, __global float *out,
                            unsigned int n) {

  const size_t size = get_global_size(0) / get_local_size(0);

  __local float local_sums[TILE_SIZE];

  inner_sum(in, partial_sums, local_sums, n);
  barrier(CLK_LOCAL_MEM_FENCE);

  // TODO get_global_size(0) / TILE_SIZE can be odd
  for (int i = get_global_size(0) / (TILE_SIZE * 2); i > 0; i /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);

    size_t group_i = get_group_id(0);
    if (group_i < i) {
// TODO make it atomic or do correctly
      partial_sums[group_i] += partial_sums[group_i + TILE_SIZE * i];
    }
  }

  out[0] = partial_sums[0];
}

// assume we are reducing only along the first dimension
__kernel void reduce_sum_2D(__global const float *in, __global float *out,
                            unsigned int axis_shape_size,
                            unsigned int resulted_shape_size) {

  size_t global_i = get_global_id(0);

  for (int i = 0; i < axis_shape_size; i++) {
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_i < resulted_shape_size) {
      out[global_i] += in[i * resulted_shape_size + global_i];
    }
  }
}
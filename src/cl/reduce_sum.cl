#define TILE_SIZE 32

// bufferSize should be power of 2 (bufferSize = next_power_of_2(numElements))
__kernel void reduce_sum_1D(__global const float *input, __global float *buffer,
                            __global float *out, unsigned int numElements,
                            unsigned int bufferSize) {
  size_t global_i = get_global_id(0);

  buffer[global_i] = (global_i < numElements) ? input[global_i] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (unsigned int s = bufferSize / 2; s > 0; s >>= 1) {
    if (global_i == 0) {
      printf("-----------------------------------------------------------------"
             "-\n");
      printf("s = %d\n", s);
    }
    printf("buffer[%ld] = %f\n", global_i, buffer[global_i]);

    if (global_i < s) {
      buffer[global_i] += buffer[global_i + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (global_i == 0) {
    out[0] = buffer[0];
  }
  return;
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

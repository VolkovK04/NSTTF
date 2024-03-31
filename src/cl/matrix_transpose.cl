#define BLOCK_DIM 16

__kernel void matrix_transpose(__global float *input, __global float *output,
                               const unsigned int width,
                               const unsigned int height) {

  __local float tile[BLOCK_DIM][BLOCK_DIM + 1];

  int xIndex = get_global_id(0);
  int yIndex = get_global_id(1);

  int transposedIndex = yIndex * width + xIndex;

  if (xIndex < width && yIndex < height) {
    tile[get_local_id(1)][get_local_id(0)] = input[transposedIndex];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  int xIndexTransposed = get_local_id(1);
  int yIndexTransposed = get_local_id(0);

  int originalIndex = yIndexTransposed * width + xIndexTransposed;

  if (xIndexTransposed < height && yIndexTransposed < width) {
    output[originalIndex] = tile[get_local_id(0)][get_local_id(1)];
  }
}
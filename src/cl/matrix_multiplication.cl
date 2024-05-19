#line 6

#define TILE_SIZE 16

__kernel void
matrix_multiplication_updated(__global const float *a, __global const float *b,
                              __global float *c, const unsigned int M,
                              const unsigned int K, const unsigned int N) {
  size_t i = get_global_id(1);
  size_t j = get_global_id(0);

  size_t local_i = get_local_id(1);
  size_t local_j = get_local_id(0);

  __local float local_a[TILE_SIZE][TILE_SIZE];

  __local float local_b[TILE_SIZE][TILE_SIZE];

  float sum = 0;

  for (size_t step = 0; step * TILE_SIZE < K; ++step) {

    if (i < M && j < K) {
      local_a[local_i][local_j] = a[i * K + local_j + step * TILE_SIZE];
    } else {
      local_a[local_i][local_j] = 0;
    }

    if (i < K && j < N) {
      local_b[local_i][local_j] = b[(local_i + step * TILE_SIZE) * N + j];
    } else {
      local_b[local_i][local_j] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef DEBUG
    if (i < M && j < K) {
      printf("local_a[%d][%d] = %f\n", local_i, local_j,
             local_a[local_i][local_j]);
    }
    if (i < K && j < N) {
      printf("local_b[%d][%d] = %f\n", local_i, local_j,
             local_b[local_i][local_j]);
    }
#endif

    for (size_t index = 0; index < TILE_SIZE; ++index) {
      sum += local_a[local_i][index] * local_b[index][local_j];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (i < M && j < N) {
    c[i * N + j] = sum;
  }
}

// (M, K, N) x (M, N, L) -> (M, K, L)
__kernel void matrix_multiplication_full(
    __global const float *a, __global const float *b, __global float *c,
    const unsigned int K, const unsigned int N, const unsigned int L,
    __global const unsigned int *shape, const unsigned int shape_size) {

  size_t i = get_global_id(0);

  size_t global_size = 1;
  if (i < shape_size) {
    global_size *= shape[i];
  }

  size_t local_size_a = global_size / (K * N);
  size_t local_size_b = global_size / (N * L);
  // local_size_a = local_size_b (should be equal)

  for (size_t j = 0; j < local_size_a; j++) {
    matrix_multiplication_updated(a + j * K * N, b + j * N * L, c + j * K * L,
                                  K, N, L);
  }
}
#define TILE_SIZE 128

__kernel void reduce_sum(__global const float *a, __global float *c,
                         const unsigned int axis, __global const size_t *shape,
                         const unsigned int shapes_size) {
  int i = get_global_id(0);

  size_t global_size = 1;
  if (i < shapes_size) {
    global_size *= shape[i];
  }

  size_t local_size = global_size / shape[axis];

  float sum = 0.0f;

  for (size_t j = 0; j < local_size; j++) {
    sum += a[j * local_size + i];
  }
}

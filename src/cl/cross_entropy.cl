__kernel void cross_entropy_loss(__global const float *predictions,
                                    __global const int *labels,
                                    __global float *loss,
                                    const int num_classes) {
  int i = get_global_id(0);

  if (i < num_classes) {
    float pred = predictions[i * num_classes + labels[i]];
    loss[i] = -log(pred);
  }
}
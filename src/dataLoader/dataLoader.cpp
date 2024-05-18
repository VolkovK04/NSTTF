#include "dataLoader.h"

namespace NSTTF {
// TODO
MNIST_DataLoader::MNIST_DataLoader() {
  // load mnist dataset on disk (if it not exist) or connect to it
  throw std::runtime_error("not implemented yet");
}

// TODO
size_t MNIST_DataLoader::size() const {
  // return count of avalible samples
  throw std::runtime_error("not implemented yet");
}

// TODO
const TensorMap &MNIST_DataLoader::operator[](size_t index) const {
  // if $(out of range)
  // throw std::out_of_range("index out of range");
  // lazy return sample by index
  // sample = TensorMap{{data: Tensor}, {lable: Tensor}}
  // data is the tensor whis shape = (image_width, image_height)
  // label is the one hot vector (tensor) with shape = (10)
  throw std::runtime_error("not implemented yet");
}

} // namespace NSTTF
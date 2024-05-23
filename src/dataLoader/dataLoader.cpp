#include "dataLoader.h"
#include "filesystem"

namespace NSTTF {
// TODO
MNIST_DataLoader::MNIST_DataLoader(const std::string &type) {
  dataset->images = read_mnist_image_file(type + "-images.idx3-ubyte");
  dataset->labels = read_mnist_label_file(type + "-labels.idx1-ubyte");
  dataset_size = dataset->labels.size();
}

size_t MNIST_DataLoader::size() const { return dataset_size; }

TensorMap MNIST_DataLoader::operator[](size_t index) const {
  if (index >= dataset_size) {
    throw std::out_of_range("index out of range");
  }
  std::cout << index << std::endl;

  size_t shape = 28;
  std::vector<float> data_vector(shape * shape);
  for (size_t i = 0; i < shape * shape; ++i) {
    data_vector.at(i) =
        ((float)dataset->images.at(shape * shape * index + i) / 255.0);
  }
  // std::vector<std::vector<size_t>> sh(shape);
  Tensor data_tensor(data_vector, std::vector<size_t>{shape, shape});

  std::vector<float> lable_vector(10);
  for (size_t i = 0; i < 10; ++i) {
    if (i == (int)dataset->labels.at(index)) {
      lable_vector.at(i) = 1.0;
      break;
    }
  }
  Tensor lable_tensor(lable_vector, {10});

  TensorMap tensorMap =
      TensorMap{{"data", data_tensor}, {"lable", lable_tensor}};

  return tensorMap;
}

uint32_t MNIST_DataLoader::read_header(const std::vector<char> &buffer,
                                       size_t position) {
  uint32_t *header = (uint32_t *)(buffer.data());

  uint32_t value = *(header + position);
  return (value << 24) | ((value << 8) & 0x00FF0000) |
         ((value >> 8) & 0X0000FF00) | (value >> 24);
}

std::vector<char> MNIST_DataLoader::read_mnist_file(const std::string &name,
                                                    uint32_t key) {
  std::filesystem::path sourcePath(_PROJECT_SOURCE_DIR);
  sourcePath.append("train_data/mnist/" + name);

  std::ifstream file;
  file.open(sourcePath, std::ios::in | std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file!");
  }
  auto size = file.tellg();
  // std::cout << size << std::endl;
  std::vector<char> buffer(size);

  // Read the entire file at once
  file.seekg(0, std::ios::beg);
  file.read(buffer.data(), size);
  file.close();

  auto magic = read_header(buffer, 0);

  if (magic != key) {
    throw std::runtime_error("Invalid magic number, probably not a MNIST file");
    return {};
  }

  auto count = read_header(buffer, 1);
  if (magic == 0x803) {
    auto rows = read_header(buffer, 2);
    auto columns = read_header(buffer, 3);

    if (size < count * rows * columns + 16) {
      std::cout << "The file is not large enough to hold all the data, "
                   "probably corrupted"
                << std::endl;
      return {};
    }
  } else if (magic == 0x801) {
    if (size < count + 8) {
      std::cout << "The file is not large enough to hold all the data, "
                   "probably corrupted"
                << std::endl;
      return {};
    }
  }
  return buffer;
}

std::vector<uint8_t>
MNIST_DataLoader::read_mnist_label_file(const std::string &name) {

  auto buffer = read_mnist_file(name, 0x801);

  if (buffer.size()) {
    auto count = read_header(buffer, 1);

    // Skip the header
    // Cast to unsigned char is necessary cause signedness of char is
    // platform-specific
    auto label_buffer = reinterpret_cast<unsigned char *>(buffer.data() + 8);

    std::vector<uint8_t> labels(count);

    for (size_t i = 0; i < count; ++i) {
      auto label = *label_buffer++;
      labels[i] = static_cast<uint8_t>(label);
    }

    return labels;
  }

  return {};
}

std::vector<uint8_t>
MNIST_DataLoader::read_mnist_image_file(const std::string &name) {
  std::vector<char> buffer = read_mnist_file(name, 0x803);

  if (buffer.size()) {
    uint32_t count = read_header(buffer, 1);
    uint32_t rows = read_header(buffer, 2);
    uint32_t columns = read_header(buffer, 3);

    // std::cout << count << std::endl
    //           << rows << std::endl
    //           << columns << std::endl;
    // Skip the header
    // Cast to unsigned char is necessary cause signedness of char is
    // platform-specific
    // uint8_t *image_buffer = reinterpret_cast<uint8_t *>(buffer.data() +
    // 16);

    std::vector<uint8_t> images(count * rows * columns);
    for (size_t i = 0; i < count * rows * columns; ++i) {
      // images.emplace_back(rows * columns);
      uint8_t pixel = buffer[i + 16];
      // std::cout << i << std::endl;
      // images[i] = static_cast<uint8_t>(pixel);
      images[i] = (uint8_t)(pixel);
    }
    return images;
  }
  return {};
}

} // namespace NSTTF
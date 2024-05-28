#include "gtest/gtest.h"

#include <dataLoader/dataLoader.h>
#include <operations/function.h>
#include <tensor/tensor.h>
#include <vector>

using namespace NSTTF;

TEST(DataLoaderTest, baseSizeCheck) {
  // Load MNIST data;
  MNIST_DataLoader train_data_loader("train");
  MNIST_DataLoader test_data_loader("test");

  EXPECT_EQ(47040000, train_data_loader.get_images().size());
  EXPECT_EQ(60000, train_data_loader.get_labels().size());
  EXPECT_EQ(7840000, test_data_loader.get_images().size());
  EXPECT_EQ(10000, test_data_loader.get_labels().size());
}

TEST(DataLoaderTest, operator) {
  // Load MNIST data;
  MNIST_DataLoader train_data_loader("train");
  MNIST_DataLoader test_data_loader("test");

  EXPECT_THROW(train_data_loader[60000], std::out_of_range);
  EXPECT_THROW(test_data_loader[10000], std::out_of_range);

  try {
    train_data_loader[59999];
  } catch (...) {
    FAIL();
  }
}

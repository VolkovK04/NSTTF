#include "../src/tensor/tensor.h"
#include "../src/utils/functions.h"
#include "gtest/gtest.h"
#include <vector>


using namespace NSTTF;

TEST(SumTest, WrongSize1) {
    Tensor a(std::vector<float>{3});
    Tensor b(std::vector<float>{5, 6});

    std::vector<Tensor> tensors;
    tensors.push_back(a);
    tensors.push_back(b);

    EXPECT_THROW({
        sum(tensors);
    }, std::runtime_error);
}

// TEST(SumTest, WrongSize2) {
//     Tensor a(std::vector<float>{2, 3});
//     Tensor b(std::vector<float>{1, 5, 6});
// }

// TEST(SumTest, WrongSize3) {
//     Tensor a(std::vector<float>{2, 3});
//     Tensor b(std::vector<float>{5, 6});
// }
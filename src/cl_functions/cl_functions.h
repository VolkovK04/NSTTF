#pragma once
#include <cstdint>

extern const char matrix_multiplication_kernel[];
extern const char matrix_transpose_kernel[];
extern const char multiplication_kernel[];
extern const char subtraction_kernel[];
extern const char sum_kernel[];

extern std::size_t matrix_multiplication_kernel_length;
extern std::size_t matrix_transpose_kernel_length;
extern std::size_t multiplication_kernel_length;
extern std::size_t subtraction_kernel_length;
extern std::size_t sum_kernel_length;
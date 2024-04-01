#include "utils.h"

namespace NSTTF {
size_t getSize(const std::vector<size_t>& shape) {
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    return size;
}
} // namespace NSTTF
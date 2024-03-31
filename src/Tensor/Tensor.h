#include <vector>

class AbstractDataPointer
{
public:
    AbstractDataPointer() = default;
    virtual ~AbstractDataPointer() = default;
    virtual void* get() = 0;
};



class Tensor
{
private:
    AbstractDataPointer* pointer = nullptr;
    const std::vector<const size_t> shape;

public:
    Tensor() = default;

    Tensor(AbstractDataPointer* pointer);

    ~Tensor() = default;
};


class AbstractPointer
{
public:
    AbstractPointer() = default;
    virtual ~AbstractPointer() = default;
    virtual void* get() = 0;
};



class Tensor
{
private:
    AbstractPointer* pointer = nullptr;
    

public:
    Tensor() = default;

    Tensor(AbstractPointer* pointer);

    ~Tensor() = default;
};
#pragma once

namespace NSTTF
{

    class AbstractOperation
    {
    public:
        AbstractOperation() = default;
        virtual ~AbstractOperation() = default;
        virtual void execute() = 0;
    };

}
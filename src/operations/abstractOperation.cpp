#include "abstractOperation.h"

namespace NSTTF
{
    const std::string AbstractOperation::getName() { return name; }
    AbstractOperation::AbstractOperation(std::string name) : name(name) {}
    UnaryOperation::UnaryOperation(std::string name, std::shared_ptr<Expression> expression) : expression(expression), AbstractOperation(name)
    {
    }
    bool UnaryOperation::equals(const Expression &expression) const
    {
        return this->expression->equals(expression);
    }
    BinaryOperation::BinaryOperation(std::string name, const Expression &left, const Expression &right) : left(left), right(right), AbstractOperation(name)
    {
    }
    bool BinaryOperation::equals(const Expression &expression) const
    {
        return this->left->equals(expression) && this->right->equals(expression);
    }
    Variable::Variable(std::string name) : AbstractOperation(name)
    {
    }

    Expression Variable::getDerivative(const Expression &expression)
    {
        if (*this == expression)
        {
            return Constant(1);
        }
    }

    std::string Variable::getName() const
    {
        return std::move(name);
    }

    bool Variable::equals(const Expression &expression) const
    {
        const Variable *var = dynamic_cast<const Variable *>(&expression);
        if (var)
        {
            return this->getName() == var->getName();
        }
        return false;
    }

    bool Expression::operator==(const Expression &other)
    {
        return this->equals(other);
    }

    bool Expression::operator!=(const Expression &other)
    {
        return !(this->equals(other));
    }

    Constant::Constant(size_t value) : value(value), AbstractOperation(std::to_string(value))
    {
    }

    Expression Constant::getDerivative(const Expression &expression)
    {
        return Constant(0);
    }

    bool Constant::equals(const Expression &expression) const
    {
        const Constant *constant = dynamic_cast<const Constant *>(&expression);
        if (constant)
        {
            return this->value == constant->value;
        }
        return false;
    }

    Sum::Sum(std::string name, const Expression &left, const Expression &right) : left(left), right(right), AbstractOperation(name)
    {
    }

    Expression Sum::getDerivative(const Expression &expression)
    {
        return Sum("Sum", left.getDerivative(expression), right.getDerivative(expression));
    }

    Subtraction::Subtraction(std::string name, const Expression &left, const Expression &right) : left(left), right(right), AbstractOperation(name)
    {
    }

    Expression Subtraction::getDerivative(const Expression &expression)
    {
        return Subtraction("Subtraction", left.getDerivative(expression), right.getDerivative(expression));
    }

    Multiplication::Multiplication(std::string name, const Expression &left, const Expression &right) : left(left), right(right), AbstractOperation(name)
    {
    }

    Expression Multiplication::getDerivative(const Expression &expression)
    {
        return Sum("Sum", Multiplication("Multiplication", left.getDerivative(expression), right), Multiplication("Multiplication", left, right.getDerivative(expression)));
    }

} // namespace NSTTF
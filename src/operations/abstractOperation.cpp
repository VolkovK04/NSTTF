#include "abstractOperation.h"

namespace NSTTF
{
    const std::string AbstractOperation::getName() { return name; }
    AbstractOperation::AbstractOperation(std::string name) : name(name) {}
    // UnaryOperation::UnaryOperation(std::string name, std::shared_ptr<Expression> expression) : expression(expression), AbstractOperation(name)
    // {
    // }
    // bool UnaryOperation::equals(std::shared_ptr<Expression> expression) const
    // {
    //     return this->expression->equals(expression);
    // }
    // BinaryOperation::BinaryOperation(std::string name, std::shared_ptr<Expression> left, std::shared_ptr<Expression> right) : left_(left), right_(right), AbstractOperation(name)
    // {
    // }
    // bool BinaryOperation::equals(std::shared_ptr<Expression> expression) const
    // {
    //     return this->left_->equals(expression) && this->right_->equals(expression);
    // }
    // Variable::Variable(std::string name) : AbstractOperation(name)
    // {
    // }
    // bool Variable::equals(std::shared_ptr<Expression> expression) const
    // {
    //     std::shared_ptr<Variable> var = std::dynamic_pointer_cast<Variable>(expression);
    //     if (var)
    //     {
    //         return this->getName() == var->getName();
    //     }
    //     return false;
    // }

    // std::shared_ptr<Expression> Variable::getDerivative(std::shared_ptr<Expression> expression) const
    // {
    //     if (this->equals(expression))
    //     {
    //         return std::make_shared<Constant>(1);
    //     }
    //     else
    //     {
    //         return std::make_shared<Constant>(0);
    //     }
    // }

    // std::string Variable::getName() const
    // {
    //     return std::move(name);
    // }

    // bool Expression::operator==(std::shared_ptr<Expression> other)
    // {
    //     return this->equals(other);
    // }

    // bool Expression::operator!=(std::shared_ptr<Expression> other)
    // {
    //     return !(this->equals(other));
    // }

    // Constant::Constant(size_t value) : value(value), AbstractOperation(std::to_string(value)) {}


    // std::shared_ptr<Expression> Constant::getDerivative(std::shared_ptr<Expression> expression) const
    // {
    //     return std::make_shared<Constant>(0);
    // }

    // bool Constant::equals(std::shared_ptr<Expression> expression) const
    // {
    //     std::shared_ptr<Constant> constant = std::dynamic_pointer_cast<Constant>(expression);
    //     if (constant)
    //     {
    //         return this->value == constant->value;
    //     }
    //     return false;
    // }

    // Sum::Sum(std::string name, std::shared_ptr<Expression> left, std::shared_ptr<Expression> right) : BinaryOperation(name, left, right)
    // {
    // }

    // std::shared_ptr<Expression> Sum::getDerivative(std::shared_ptr<Expression> expression) const
    // {
    //     return std::make_shared<Sum>("Sum", left_->getDerivative(expression), right_->getDerivative(expression));
    // }

    // Subtraction::Subtraction(std::string name, std::shared_ptr<Expression> left, std::shared_ptr<Expression> right) : BinaryOperation(name, left, right)
    // {
    // }

    // std::shared_ptr<Expression> Subtraction::getDerivative(std::shared_ptr<Expression> expression) const
    // {
    //     return std::make_shared<Subtraction>("Subtraction", left_->getDerivative(expression), right_->getDerivative(expression));
    // }

    // Multiplication::Multiplication(std::string name, std::shared_ptr<Expression> left, std::shared_ptr<Expression> right) : BinaryOperation(name, left, right)
    // {
    // }

    // std::shared_ptr<Expression> Multiplication::getDerivative(std::shared_ptr<Expression> expression) const
    // {
    //     return std::make_shared<Sum>("Sum", std::make_shared<Multiplication>("Multiplication", left_->getDerivative(expression), right_), std::make_shared<Multiplication>("Multiplication", left_, right_->getDerivative(expression)));
    // }

} // namespace NSTTF
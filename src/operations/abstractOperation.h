#pragma once

#include <string>
#include <tensor/tensor.h>

namespace NSTTF
{

  class Expression
  {

  public:
    Expression() = default;
    virtual Expression getDerivative(const Expression &expression) = 0;
    virtual ~Expression() = default;

    // AbstractNode asNode() const;
    // идея в том, чтобы получить дифференциал и кастануть его в ноду или во что-то иное

    virtual bool equals(const Expression &expression) const = 0;

    virtual bool operator==(const Expression &other);
    virtual bool operator!=(const Expression &other);
  };

  class AbstractOperation
  {

  protected:
    std::string name;

  public:
    const std::string getName();

    AbstractOperation() = default;
    AbstractOperation(std::string name);

    virtual ~AbstractOperation() = default;
  };

  class UnaryOperation : public Expression, public AbstractOperation
  {

  private:
    std::shared_ptr<Expression> expression;

  public:
    UnaryOperation(std::string name, std::shared_ptr<Expression> expression);

    bool equals(const Expression &expression) const override;
  };

  class BinaryOperation : public Expression, public AbstractOperation
  {

  protected:
    const Expression &left;
    const Expression &right;

  public:
    BinaryOperation(std::string name, const Expression &left, const Expression &right);

    bool equals(const Expression &expression) const override;
  };

  class Variable : public Expression, public AbstractOperation
  {

  public:
    Variable(std::string name);
    Expression getDerivative(const Expression &expression) override;
    std::string getName() const;

    bool equals(const Expression &expression) const override;
  };

  class Constant : public Expression, public AbstractOperation
  {

  private:
    size_t value; // broadcasting возможно бахнуть

  public:
    Constant(size_t value);
    Expression getDerivative(const Expression &expression) override;
    bool equals(const Expression &expression) const override;
  };

  class Sum : public AbstractOperation, public BinaryOperation
  {

  public:
    Sum(std::string name, const Expression &left, const Expression &right);
    Expression getDerivative(const Expression &expression) override;
  };

  class Subtraction : public AbstractOperation, public BinaryOperation
  {
  public:
    Subtraction(std::string name, const Expression &left, const Expression &right);
    Expression getDerivative(const Expression &expression) override;
  };

  class Multiplication : public AbstractOperation, public BinaryOperation
  {
  public:
    Multiplication(std::string name, const Expression &left, const Expression &right);
    Expression getDerivative(const Expression &expression) override;
  };

  class Division : public AbstractOperation, public BinaryOperation
  {
  public:
    // Можно не надо?
    // Division(std::string name, const Expression& left, const Expression& right);
    // Expression getDerivative(const Expression& expression) override;
  };

  class Exponent : public AbstractOperation, public BinaryOperation
  {
  public:
    // Можно не надо?
    // Exponent(std::string name, const Expression& left, const Expression& right);
    // Expression getDerivative(const Expression& expression) override;
  };

} // namespace NSTTF
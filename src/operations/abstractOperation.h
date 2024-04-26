#pragma once

#include <string>
#include <memory>

namespace NSTTF
{

  class Expression
  {

  public:
    Expression() = default;
    virtual std::shared_ptr<Expression> getDerivative(std::shared_ptr<Expression> expression) const = 0;
    virtual ~Expression() = default;

    // AbstractNode asNode() const;
    // идея в том, чтобы получить дифференциал и кастануть его в ноду или во что-то иное

    virtual bool equals(std::shared_ptr<Expression> expression) const = 0;

    virtual bool operator==(std::shared_ptr<Expression> other);
    virtual bool operator!=(std::shared_ptr<Expression> other);
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

    bool equals(std::shared_ptr<Expression> expression) const override;
  };

  class BinaryOperation : public Expression, public AbstractOperation
  {

  protected:
    std::shared_ptr<Expression> left_;
    std::shared_ptr<Expression> right_;

  public:
    BinaryOperation(std::string name, std::shared_ptr<Expression> left, std::shared_ptr<Expression> right);

    bool equals(std::shared_ptr<Expression> expression) const override;
  };

  class Variable : public Expression, public AbstractOperation
  {

  public:
    Variable(std::string name);
    std::shared_ptr<Expression> getDerivative(std::shared_ptr<Expression> expression) const override;
    std::string getName() const;

    bool equals(std::shared_ptr<Expression> expression) const override;
  };

  class Constant : public Expression, public AbstractOperation
  {

  private:
    size_t value; // broadcasting возможно бахнуть

  public:
    Constant(size_t value);
    std::shared_ptr<Expression> getDerivative(std::shared_ptr<Expression> expression) const override;
    bool equals(std::shared_ptr<Expression> expression) const override;
  };

  class Sum : public BinaryOperation
  {

  public:
    Sum(std::string name, std::shared_ptr<Expression> left, std::shared_ptr<Expression> right);
    std::shared_ptr<Expression> getDerivative(std::shared_ptr<Expression> expression) const override;
  };

  class Subtraction : public BinaryOperation
  {
  public:
    Subtraction(std::string name, std::shared_ptr<Expression> left, std::shared_ptr<Expression> right);
    std::shared_ptr<Expression> getDerivative(std::shared_ptr<Expression> expression) const override;
  };

  class Multiplication : public BinaryOperation
  {
  public:
    Multiplication(std::string name, std::shared_ptr<Expression> left, std::shared_ptr<Expression> right);
    std::shared_ptr<Expression> getDerivative(std::shared_ptr<Expression> expression) const override;
  };

  class Division : public BinaryOperation
  {
  public:
    // Можно не надо?
    // Division(std::string name, std::shared_ptr<Expression> left, std::shared_ptr<Expression> right);
    // Expression getDerivative(std::shared_ptr<Expression> expression) override;
  };

  class Exponent : public BinaryOperation
  {
  public:
    // Можно не надо?
    // Exponent(std::string name, std::shared_ptr<Expression> left, std::shared_ptr<Expression> right);
    // Expression getDerivative(std::shared_ptr<Expression> expression) override;
  };

} // namespace NSTTF
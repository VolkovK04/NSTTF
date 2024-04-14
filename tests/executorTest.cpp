#include "gtest/gtest.h"

#include "../src/computationGraph/computationGraph.h"
#include "../src/utils/compiler.h"
#include "../src/utils/graphExecutor.h"

#include <map>

using namespace NSTTF;

TEST(GraphTests, SumNode) {
  Compiler compiler;
  ComputationGraph g;
  std::map<std::string, Tensor> tensorsMap = {
      {"test1", Tensor{{1.f, 2.f, 3.f}, {1, 3}}},
      {"test2", Tensor{{4.f, 5.f, 6.f}, {1, 3}}},
  };

  NodeInterface nodeInterface1 = g.AddInputNode("test1");
  NodeInterface nodeInterface2 = g.AddInputNode("test2");
  NodeInterface sumNode = nodeInterface1 + nodeInterface2;
  sumNode.setOutput();

  GraphExecutor executor = compiler.compile(g);
  std::map<std::string, Tensor> m = executor.execute(tensorsMap);

  std::map<std::string, Tensor>
}

TEST(GraphTests, SubNode) {
  ComputationGraph g;
  NodeInterface nodeInterface1 = g.AddInputNode("test1");
  NodeInterface nodeInterface2 = g.AddInputNode("test2");
  NodeInterface sumNode = nodeInterface1 - nodeInterface2;
  const OperationNode *node =
      static_cast<const OperationNode *>(&sumNode.getNode());
  EXPECT_EQ(node->getOperation().getName(), "subtraction");
}

TEST(GraphTests, MultNode) {
  ComputationGraph g;
  NodeInterface nodeInterface1 = g.AddInputNode("test1");
  NodeInterface nodeInterface2 = g.AddInputNode("test2");
  NodeInterface sumNode = nodeInterface1 * nodeInterface2;
  const OperationNode *node =
      static_cast<const OperationNode *>(&sumNode.getNode());
  EXPECT_EQ(node->getOperation().getName(), "multiplication");
}
#include "gtest/gtest.h"
#include <computationGraph/computationGraph.h>

using namespace NSTTF;

TEST(GraphTests, CreateTest) {
    ComputationGraph g;
    EXPECT_EQ(g.getInputNodes().size(), 0);
    EXPECT_EQ(g.getOutputNodes().size(), 0);
}

TEST(GraphTests, CreateInputNode1) {
    ComputationGraph g;
    NodeInterface nodeInterface = g.AddInputNode("test");

    EXPECT_EQ(g.getInputNodes().size(), 1);
    EXPECT_EQ(g.getOutputNodes().size(), 0);

    EXPECT_EQ(nodeInterface.getNode().getName(), "test");
}

TEST(GraphTests, CreateInputNode2) {
    ComputationGraph g;
    NodeInterface nodeInterface1 = g.AddInputNode("test1");
    NodeInterface nodeInterface2 = g.AddInputNode("test2");
    EXPECT_EQ(g.getInputNodes().size(), 2);
}


TEST(GraphTests, SumNode) {
    ComputationGraph g;
    NodeInterface nodeInterface1 = g.AddInputNode("test1");
    NodeInterface nodeInterface2 = g.AddInputNode("test2");
    NodeInterface sumNode = nodeInterface1 + nodeInterface2;
    sumNode.setOutput();
    EXPECT_EQ(g.getInputNodes().size(), 2);
    EXPECT_EQ(g.getOutputNodes().size(), 1);
    const OperationNode* node = static_cast<const OperationNode*>(&sumNode.getNode());
    EXPECT_EQ(node->getOperation().getName(), "sum");    
}

TEST(GraphTests, SubNode) {
    ComputationGraph g;
    NodeInterface nodeInterface1 = g.AddInputNode("test1");
    NodeInterface nodeInterface2 = g.AddInputNode("test2");
    NodeInterface sumNode = nodeInterface1 - nodeInterface2;
    const OperationNode* node = static_cast<const OperationNode*>(&sumNode.getNode());
    EXPECT_EQ(node->getOperation().getName(), "subtraction");    
}

TEST(GraphTests, MultNode) {
    ComputationGraph g;
    NodeInterface nodeInterface1 = g.AddInputNode("test1");
    NodeInterface nodeInterface2 = g.AddInputNode("test2");
    NodeInterface sumNode = nodeInterface1 * nodeInterface2;
    const OperationNode* node = static_cast<const OperationNode*>(&sumNode.getNode());
    EXPECT_EQ(node->getOperation().getName(), "multiplication");    
}

TEST(GraphTests, MatrixMultNode) {
    ComputationGraph g;
    NodeInterface nodeInterface1 = g.AddInputNode("test1");
    NodeInterface nodeInterface2 = g.AddInputNode("test2");
    NodeInterface multNode = NodeInterface::MatrixMult(nodeInterface1, nodeInterface2);
    const OperationNode* node = static_cast<const OperationNode*>(&multNode.getNode());
    EXPECT_EQ(node->getOperation().getName(), "matrix_multiplication");    
}

TEST(GraphTests, MatrixTranspose) {
    ComputationGraph g;
    NodeInterface nodeInterface = g.AddInputNode("test");
    NodeInterface trNode = NodeInterface::MatrixTranspose(nodeInterface);
    const OperationNode* node = static_cast<const OperationNode*>(&trNode.getNode());
    EXPECT_EQ(node->getOperation().getName(), "matrix_transpose");    
}

TEST(GraphTests, NodesFromDifferentGraphs) {
  try {
    ComputationGraph g1;
    NodeInterface node1 = g1.AddInputNode("node1");
    ComputationGraph g2;
    NodeInterface node2 = g2.AddInputNode("node2");
    NodeInterface node3 = node1 + node2;

    FAIL() << "No exception";
  } catch (std::runtime_error &err) {
    EXPECT_EQ(err.what(), std::string("Nodes defined in different graphs"));
  } catch (...) {
    FAIL() << "Another exception";
  }
}
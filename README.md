# NSTTF Plan

## 1) API structure (methods, components etc.) development
- Develop our library's API structure based on your goals and scope.
- Define the classes and functions that will be available to users of our library.
(_approximately by 11.03.24 as part of the second milestone_)

## 2) Implementation of basic data structures (Writing Tensor and connected with it classes)
- Create classes to represent graph nodes, operations, connections between nodes, etc.
- Define basic methods for working with these data structures, such as adding and removing nodes, performing operations, etc.

## 3) Implementation of tree evaluation, based on structures above

## 4) Implementation of specific optimizations and resource distributions for tree evaluation
- Develop algorithms to optimize and perform operations on a computational graph.
- Implement methods to allocate resources and control the execution of operations on various devices (e.g. CPU, GPU).
(_work starts after the second milestone. Probably we have time and some kind of prototype will be prepared_)

## 5) Cover logic with tests for implemented above (during the whole development)
(_this point is the one the others is going to be tested on_)

---

### What will be ready by the second milestone in brief (all details are mentioned above):
(_Criterias for each milestone_)

## Second milestone (01.04.24):
1. API structure
2. Prototype of basic data structures with tests
   - Tensors
   - Operations
   - Dependencies and connections
   - Prototype of data distribution
   - Tests (sum, sub, mult, conc)
3. Prototype of evaluation tree with tests
   - Building evaluation trees on given equations (just prototype based on API)

## Third milestone (29.04.24):
1. Extended eval tree:
   - Building derivative equation based on tree
   - Tests (creation, building and evaluation)
2. GPU optimizations prototype development
   - Parallel evaluation of tensors with OpenCL 
   - Lazy evaluation and improving data distribution 
   - All-reduce
   - Tests (data distribution based on GPU)

## Fourth milestone (27.05.24):
1. Polishing base structure
2. Adding more logic with test coverage if needed
3. Implementation of basic layers for training
   _GOAL: Training if a simple NN should work_
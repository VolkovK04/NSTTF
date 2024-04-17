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
By the third phase:
1. Construction of a differentiation graph
   - Implement the backpropagation algorithm
   - Add flags to instruct the compiler on which nodes the algorithm should reach (or pass them as arguments)
   - Define the derivatives for all existing operations concerning all their arguments (for some, it may be necessary to write new compute language functions, while for others, existing ones can be reused)
2. Implement the gradient descent algorithm (GD/SGD)
   - Simply implement the following operation:
     ```
     W = W - learning_rate * grad(W)
     ```
   - If there is time and interest, you could also implement some of the following algorithms: Momentum, Nesterov Momentum, RMSProp, Adam, AdamMax
3. Write a TensorDict class that acts as a wrapper over
   `std::map<std::string name, Tensor tensor>` and overloads all operations defined over tensors
   - Operations that take more than one tensor should check that the dictionaries contain tensors with the same names and apply operations to such pairs
4. Write functional tests
5. (?) TensorStack class representing an array of tensors of the same size
   - Most operations with such stacks will be just as efficient
   - During computations, it can be converted into a regular tensor
   - It should be possible to add and remove tensors from the stack without additional memory allocations
6. (?) Function class that contains:
   - The name of the function
   - The algorithm for its computation
   - The algorithm for its differentiation (which may differ for various arguments)
   - Some additional data, such as the number of parameters and any flags needed by the compiler
7. Efficiency tests for all operations
8. ConstNode class that initializes from a tensor and then functions like an InputNode
9. (?) Tensor broadcasting
10. (?) Provide users with tools to create their own functions (additional)
11. (?) CompositNode class that contains an entire graph within itself; the graph's inputs and outputs should link with the node's inputs and outputs. Should be supported at the compiler level

## Fourth milestone (27.05.24):
1. Implement some graph optimization algorithms
   - Constant propagation (nodes that depend only on constants can be calculated at compile time)
   - Node fusion (for example, `a * b + c` can be replaced by calling one cl function like `multAndSum(a, b, c)`)
   - Parenthesis placement when multiplying more than two matrices (solved in an algorithms course)
   - Optimization by distributivity (`AB+AC = A(B+C)`, similarly with the right side)
I imagine it like this:
   - we traverse the graph and generate naive code
   - then in a loop, we look for some patterns in this code similar to regexes and replace them with more efficient code
   - as soon as none of the patterns are found, we finish
It's similar to a string problem from the C course
2. Train a simple neural network and compare the results with the original TensorFlow
   - Data preprocessing (for mnist you will need to somehow load the dataset and convert it to a tensor)
   - Model construction (construct a graph from already written blocks)
   - Write a training loop (run a batch, calculate gradients, optimize weights)
   - Validate the model on a test sample (calculate the error and make sure everything is OK)

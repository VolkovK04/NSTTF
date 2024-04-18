# NSTTF Plan
## Stands for Not So Tiny Tensorflow (a.k.a. Tensorlow Killer)
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
1. Construction of a differentiation graph -> Yolkin Maxim
   - Implement the backpropagation algorithm
   - Add flags to instruct the compiler on which nodes the algorithm should reach (or pass them as arguments)
   - Define the derivatives for all existing operations concerning all their arguments (for some, it may be necessary to write new compute language functions, while for others, existing ones can be reused)
2. Implement the gradient descent algorithm (GD/SGD) -> Yolkin Maxim
   - Simply implement the following operation:
     ```
     W = W - learning_rate * grad(W)
     ```
   - If there is time and interest, you could also implement some of the following algorithms: Momentum, Nesterov Momentum, RMSProp, Adam, AdamMax
3. Write a TensorDict class that acts as a wrapper over -> Volkov Kirill
   `std::map<std::string name, Tensor tensor>` and overloads all operations defined over tensors
   - Operations that take more than one tensor should check that the dictionaries contain tensors with the same names and apply operations to such pairs
4. Write functional tests -> Ovchinnikov Maksim
5. (?) TensorStack class representing an array of tensors of the same size
   - Most operations with such stacks will be just as efficient
   - During computations, it can be converted into a regular tensor
   - It should be possible to add and remove tensors from the stack without additional memory allocations
6. (?) Function class that contains: -> Novikov Serega
   - The name of the function
   - The algorithm for its computation
   - The algorithm for its differentiation (which may differ for various arguments)
   - Some additional data, such as the number of parameters and any flags needed by the compiler
7. Efficiency tests for all operations -> Ovchinnikov Maksim
8. ConstNode class that initializes from a tensor and then functions like an InputNode -> Volkov Kirill
9. (?) Tensor broadcasting -> Ovchinnikov Maksim
10. (?) Provide users with tools to create their own functions (additional)
11. (?) CompositNode class that contains an entire graph within itself; the graph's inputs and outputs should link with the node's inputs and outputs. Should be supported at the compiler level
12. (?) Code documentation
13. CL code to hex array (function in utils for that) -> Novikov Serega

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

# andrew-ng-improving-deep-neural-networks
My notes, the most useful code snippets, and an archive of my jupyter notebook HW assignments for [Andrew Ng's excellent course](https://www.coursera.org/learn/deep-neural-network/home/welcome).

## Notebooks
The file [improving-deep-neural-networks-course-notebook](improving-deep-neural-networks-course-notebook.ipynb) includes all my notes that I took while watching the lecture videos (along with screen-capping what I considered the most useful images / slides from his powerpoints).

The homework notebooks and files (organized by week) and a tarred archive of the full jupyter workspace for the class are stored in the folder [hw-notebooks](hw-notebooks).

## Code Base

**Initialization** in [initalization_functions.py](code-base/initialization_functions.py)
- Just implements a function for "He Initialization" which multiples the random weights by $\sqrt{\frac{2}{\text{dimension of the previous layer}}}$ (for RelU units) to keep them appropriately small in proportion to the number of input features from the underlying node.

**Regularization** in [regularization_functions.py](code-base/regularization_functions.py)
- A wrapper model function that allows for L2 regularization (`lambd` kwarg) OR dropout (`keep_prob` kwarg)
- Helper functions that modify cost computation and backprop for the case of L2 regularization.
- Helper functions that modify forward prop and backprop for the case of dropout.
    
**Gradient Checking** in [gradient_check_functions.py](code-base/gradient_check_functions.py)
- The wrapper function relies on three helper functions: 
    - `parameter_values = dictionary_to_vector(parameters)` takes in your dictionary of parameters and unravels then concatenates them into one long vector
    - `parameters = vector_to_dictionary(parameter_values)` reverses this process
    - `grad = gradients_to_vector(gradients)` performs the same process on the dictionary of gradients
- `difference = gradient_check_n(parameters, gradients, X, Y, epsilon)` calls the two above vector-making helper functions and then loops through the elements of parameter_values. For each parameter value it $\pm$ increments only that value and calls `forward_propagation` to get $J_+$ and $J_-$ for computing the approximate partial derivative, which is added to the vector `gradapprox`. After the loop `gradapprox` is compared to `grad` and the difference (as defined in lecture notes) is returned.

**Optimization** in [optimization_functions.py](code-base/optimization_functions.py)
- Wrapper model function to perform mini-batch GD using standard GD, GD with momentum, or ADAM. 
- Helper function to create random minibatches from X and y
- Helper functions to initialize velocity variables and perform parameter update with the momentum optimizer.
- Helper functions to initialize velocity and RMSprop variables and perform parameter update with the ADAM optimizer

**Multiclass Classification with Tensorflow** in [tensorflow_functions.py](code-base/tensorflow_functions.py)
- Wrapper model function that defines the computational graph (3 layer NN using a softmax output layer for 6-class classification), initializes the tensors in the graph, runs the session on every iteration of mini-batch GD.
- Helper functions to:
	- Create TF placeholders for X, y data inputs
	- Create TF variables for NN parameters
	- Define forward prop computations with parameters and input
	- Compute the cost from the output of the forward prop
# andrew-ng-neural-nets-and-deep-learning
My notes, the most useful code snippets, and an archive of my jupyter notebook HW assignments for [Andrew Ng's excellent course](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome).

## Notebooks
The file [neural-nets-and-deep-learning-course-notebook](neural-nets-and-deep-learning-course-notebook.ipynb) includes all my notes that I took while watching the lecture videos (along with screen-capping what I considered the most useful images / slides from his powerpoints).

The homework notebooks and files (organized by week) and a tarred archive of the full jupyter workspace for the class are stored in the folder [hw-notebooks](hw-notebooks).

## Code for Deep (L-Layer) NNs
The file [Deep_NN_Functions](code-base/L_Layer_NN_Functions.py) has been pulled out of the homework workspace and modified slightly; it contains all the helper and wrapper functions that we built in the class for constructing and running a deep L-Layer NN. The code organization is as follows:

**Initialization**: Loops through the layers to initialize NN parameters.

- `parameters = initialize_parameters_deep(layer_dims)` takes in a list of the number of "neurons" in each layer (input, hidden1..., hiddenL-1, and output) and returns a dictionary of matrices (with correct dimensions) with random initial values for the parameters $W$ and $b$ of the hidden and output layers.


**Forward Pass**: 

Each layer computes a linear transform followed by activation so these two functions are written separately and each stores a cache of values that are used by backprop.

- `Z, linear_cache = linear_forward(A_prev, W, b)` where `linear_cache: (A_prev, W, b)`. Performs the linear part of the computation of a layer to output $Z$ and cache the passed in $A$ and layer parameters.

- `A, activation_cache = relu(Z)` where `activation_cache: (Z)`. Performs the non-linear part of the computation of a layer to output $A$ and cache the passed in $Z$ value. An analogous function is defined `sigmoid(Z)`.


The linear and activation pieces of a single layer computation are combined into one function, and then a wrapper function executes the full sequential forward pass.

- `A, cache = linear_activation_forward(A_prev, W, b, activation_type)` where `cache: (linear_cache, activation_cache)`. Performs the linear + activation computation of a layer to output `A` for passing forward and to collect the two caches.

- `AL, caches = L_model_forward(X, parameters)` where `caches: [(lin_cache_1, act_cache_1)..., (lin_cache_L, act_cache_L)]`. Loops through the layers calling `linear_activation_forward`, adding the two caches into a list and passing its output of $A$ forward into the overlying layer. The loop is initialized with $A^{[0]}=X$, and the output of the last layer is `AL` which gives our predictions $\hat{Y}$.


**Cost**: This follows the same structure as the forward pass, and relies on cached values from that pass.

- `cost = compute_cost(AL, Y)` computes the cost function for the current predictions by the net (doesn't actually use `parameters` input).


**Backward Pass**: 

This follows the same structure as the forward pass, however since we are working backwards the activation component of a single layer's computation is considered *before* the linear component. These functions rely on cached values from the forward pass.

- `dZ = relu_backward(dA, activation_cache)` Receives a passed in values of `dA` computed by the overlying node and uses it to compute `dZ` (also relying on the cached value of $Z$.

- `dA_prev, dW, db = linear_backward(dZ, linear_cache)` Uses the value of `dZ` and the cached values of `W` and `A_prev` to compute derivatives. Note that the computed `dA_prev` is now available to pass backward into the next (underlying) node.

The activation and linear pieces of a single layer computation are combined into one function, and then a wrapper function executes the full sequential backward pass.

- `dA_prev, dW, db = linear_activation_backward(dA, cache, activation_type)` where `cache: (linear_cache, activation_cache)`. Performs the activation and linear components of the derivative computations of a layer to output `dA_prev` for passing backward and to collect the derivatives needed for parameter updates.

- `grads = L_model_backward(AL, Y, caches)` where `caches: [(lin_cache_1, act_cache_1)..., (lin_cache_L, act_cache_L)]`. Loops backward through the layers calling `linear_activation_backward`, adding the derivates into a dictionary and passing its output of `dA_prev` backward into the underlying layer. The loop is initialized with `dAL` which is computed from the form of the cost function.


**Parameter Updates**:

- `parameters = update_parameters(parameters, grads, learning_rate)` simply performs the update rule to return the new parameter values based on this step of GD

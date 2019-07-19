# andrew-ng-convolutional-neural-networks
My notes, the most useful code snippets, and an archive of my jupyter notebook HW assignments for [Andrew Ng's excellent course](https://www.coursera.org/learn/convolutional-neural-networks/home/welcome).

## Notebooks
The file [convolutional-neural-networks-course-notebook](convolutional-neural-networks-course-notebook.ipynb) includes all my notes that I took while watching the lecture videos (along with screen-capping what I considered the most useful images / slides from his powerpoints).

The homework notebooks and files (organized by week) and a tarred archive of the full jupyter workspace for the class are stored in the folder [hw-notebooks](hw-notebooks).

## Code Base

### [conv-basics.py](code-base/conv-basics.py)
A basic Numpy implementation of convolutional and pooling layers forward pass operations. The forward pass code is organized as follows:

**Inputs:** 
- `A_prev`: the 4D output volume of the previous layer (vectorized so that the data from all samples are stacked together)
    - Shape (m, n_H_prev, n_W_prev, n_C_prev)
- `W`: a 4D tensor consisting of all the 3D filter weight tensors for the layer (there are $n_c$ of these) stacked together
    - Shape (f, f, n_C_prev, n_C)
- `b`: a 4D tensor consisting of all the real-number filter biases for the layer stacked together
    - (1, 1, 1, n_C)
- `hparameters`: a dictionary with the stride and pad for the layer

**Outputs:**
- `Z`: the 4D tensor output of the CONV layer (before applying a the non-linear activation)
    - Shape (m, n_H, n_W, n_C)
- `cache`: Cache of the input values which is needed for back-prop (A_prev, W, b, hparameters)

**Function:**
- We initialize the output volume `Z` with zeros of the correct shape
- We zero-pad the vertical and horizontal dimensions of input volume to get `A_prev_pad`
- Loop through the samples in `A_prev_pad` (i):
    - Loop over the vertical axis of the output volume (h):
        - Loop over the horizontal axis of the output volume (w):
            - Loop over the channels i.e. depth axis of the output volume (c) <- THIS MEANS LOOP OVER THE FILTERS OF THE LAYER! 
                - Each index pair (h, w) in the output volume corresponds to a location of overlaying a filter onto the input volume i.e. defining a slice of the input volume
                - Multiply and sum this particular 3D region/slice (determined by $h$ and $w$) of the $i^{th}$ sample with the $c^{th}$ 3D filter, the result is a real number which belongs at the [i, h, w, c] index of the output volume


###[tensorflow-simple-model.py](code-base/tensorflow-simple-model.py)
A Tensorflow implementation of a simple CNN for sign language digit recognition.

As a refresher, writing and running programs in TensorFlow has the following steps:
1. Create Tensors (variables) that are not yet executed/evaluated. 
2. Write operations between those Tensors. (This defines the computational graph)
3. Initialize your Tensors. 
4. Create a Session. 
5. Run the Session. This will run the operations you'd written above. 

Some things to keep in mind:
- The two main object classes in tensorflow are Tensors and Operators. 
- When you code in tensorflow you have to take the following steps:
    - Create a graph containing Tensors (Variables, Placeholders ...) and Operations on tensors (tf.matmul, tf.add, ...)
		- A placeholder is an object whose value you can specify only later by passing in values to a session using a "feed dictionary" (`feed_dict` kwarg)
    - Create a session and then Initialize the session (initial values for tensor objects)
    - Run the session to execute the graph
        - You can execute the graph multiple times: think of the session as a block of code to train the model - each time you run the session on a minibatch, it trains the parameters.
        - The backpropagation and optimization is automatically done when running the session on the "optimizer" object.

In our vectorized version of a CNN, each sample is an RGB image (3D matrix), and all the samples in a minibatch are concatenated along the **first** dimension to form a 4D matrix which is the input to the first layer. For multiclass classification each sample has a corresponding $y$ vector where the number of components is the number of classes. We first create two `tf.placeholder`s for these input $X$ and $Y$ matrices. The first dimension corresponds to the number of samples, for which we specify a size of `None` so that different numbers of samples may be used as needed.

The filters for a given layer are likewise concatenated together along the **last** dimension to create a single 4D matrix $W$ that contains all the weights for a CONV layer. Since these are dynamic values in our graph, we use `tf.get_variable` to create them. We also specify the way their values should be initialized. Here there are 8 filters in CONV1 and 16 filters in CONV2.
    
The network itself is specified by a single function that defines the forward prop chain of computation. Note that the convolution and the non-linear activations are called as separate steps. This model follows each CONV with a max pool. Both these types of layers have stride inputs that specify a stride in every dimension of the input size (since the first dimension corresponds to the concatenation of samples, we give it stride of 1). After the two CONV->POOL segments, we flatten the output and feed it into a fully connected layer with 6 neurons; this performs only the linear part of the computation for an FC layer to get Z3, the non-linear part to get A3 is actually combined with the loss function in TF.

As mentioned, the forward prop ends with the linear computation for the FC layer which gives Z3. The non-linear part to get A3 is actually combined with the loss function in TF. To get a single cost number we average across the loss of all the samples.
    
The wrapper to fully specify and train the model calls the above helper functions to create the model, defines the optimizer to use, then initializes a session and runs it on minibatches.

    
###[keras-basic-model.py](code-base/keras-basic-model.py)
A Keras implementation of a simple CNN for happy vs. not happy face image classification.

Keras uses a different convention with layer variable names: rather than creating a new variable on each step of forward prop (e.g. Z1, A1, Z2), each computation step just reassigns X to a new value e.g. X = keras_func(kwargs)(X).

For training / testing there are four steps:
1. Create the model by specifying layer-by-layer operations
2. Compile the model by calling `model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])`
3. Train the model on train data by calling `model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)`
4. Test the model on test data by calling `model.evaluate(x = ..., y = ...)`

Two useful commands to inspect / visualize your achitecture are `myModel.summary()` which gives your layers in table form and `SVG(model_to_dot(myModel).create(prog='dot', format='svg'))` which gives a graphical representation.


###[keras-resnet-model.py](code-base/keras-resnet-model.py)
A Keras implementation of a ResNet CNN for sign language digit recognition.

An *identity block* is the standard residual block where the input ($a^{[l]}$) has the same shape as the output $a^{[l+N]}$ where $N$ is the number of skipped layers. A *convolution block* is a residual block where there is a shape mismatch between the input and output, so the shortcut path needs to involve a reshaping convolution. Note there is no non-linearity applied after the convolution on the shortcut path.


###[keras-nonmax-suppression.py](code-base/keras-nonmax-suppression.py)
A Keras implementation of extending pretrained YOLO model with non-max suppression. 

We load a pretrained YOLO model in Keras and write functions that extend the model with layers that perform non-max suppression filtering to output our final best predictions. Then we write a predict function that will run a single image through our extended graph and output the final predictions. We call our extension functions within an interactive session, and pass that same session into the predict function.


###[face-recognition-model.py](code-base/face-recognition-model.py)
An implementation of face verification and recognition tasks using a pretrained net for generating vector encodings of the image. Designed to be used with a pretrained Keras implementation of FaceNet (an inception architecture).
- Function `dist, door_open = verify(image_path, identity, database, model)` performs verification and returns the "distance" between the two encodings and whether or not you should open the door for that person
- Function `min_dist, identity = who_is_it(image_path, database, model)` performs checking against a database to find the "closest match" within a distance of 0.7 (otherwise no match is found).


###[tensorflow-neural-style-transfer.py](code-base/tensorflow-neural-style-transfer.py)
A Tensorflow implementation of neural style transfer.

We write functions to compute the content component of the cost and the style component of the cost (which is an average of the cost computed from several different layers), these are combined into one complete cost function. 
- `J = total_cost(J_content, J_style, alpha, beta)` computes the full cost by calling
    - `J_content = compute_content_cost(a_C, a_G)`
    - `J_style = compute_style_cost(model, STYLE_LAYERS)` which averages the individual layer style costs calculated with 
        - `J_style_layer = compute_layer_style_cost(a_S, a_G)`


We then perform the style transfer using a pretrained VGG model according to the steps below:
1. Create an Interactive Session
2. Load the content image 
3. Load the style image
4. Randomly initialize the image to be generated 
5. Load the VGG16 model
7. Build the TensorFlow graph with an interactive session:
    - Extend the graph to run the content image through the VGG16 model and compute the content cost
    - Extend the graph to run the style image through the VGG16 model and compute the style cost
    - Add a computation (layer) to compute the total cost
    - Define the optimizer and the learning rate
8. Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.
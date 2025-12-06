# filepath: d:\Billy's copy\machine_learning\LearningMachineLearning\Learning\michael_nielsen_deep_learning_book\chapter1_\network.py
# In this implementation, the main goal is to divide the minist data set 
# into 50, 000 training set + 10, 000 validation set. Alongside with 
# 10, 000 testing data sets.
import random
import numpy as np
import math

# First we need to create a neural network class, 
class Network(object):

    # sizes contains te number of neurons in the respective layers. For example, if we want to create a
    # Network object iwth 2 neurons in the first layer, 3 neurons in the second layer, and 1 neuron in the final layer, 
    # we would do this we code 
    # net = Network([2, 3, 1])
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[: -1], sizes[1:])]

    # This just returns the next layer
    def feedforward(self, a):
        """Return the output of the network if "a"  is input. """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        """Train the neural network using mini-batch stochastic
            gradient descent.  The "training_data" is a list of tuples
            "(x, y)" representing the training inputs and the desired
            outputs.  The other non-optional parameters are
            self-explanatory.  If "test_data" is provided then the
            network will be evaluated against the test data after each
            epoch, and partial progress printed out.  This is useful for
            tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k: k+mini_batch_size] for k in range(0, n, mini_batch_size)
                ]
            for mini_batch in mini_batches:
                # choose one of two update implementations:
                # self.update_mini_batch(mini_batch, eta)      # sample-by-sample (original)
                self.update_mini_batches_matrix_based(mini_batch, eta)  # fully matrix-based
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test
                ))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """ Update the network's weights and biases by applying
            gradient descent using backpropagation to a single mini batch.
            The "mini_batch" is a list of tuples "(x, y)", and "eta"
            is the learning rate. """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # The following two nablas are another way of saying dC/dw and db/dw for one specific training data
            # They can be thought as the gradient direction for weight and bias that will help minimize the cost function
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta/len(mini_batch)) * nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb 
                        for b, nb in zip(self.biases, nabla_b)]
    # note that for this function, most of the work is done by the line self.backprop(x, y)
    # which computes the gradient of the cost function
    # self.backprop will be implemented later in the leanring process

    def update_mini_batches_matrix_based(self, mini_batch, eta):
        """
        Fully matrix-based mini-batch update.
        This method stacks the inputs and outputs of the mini-batch into matrices
        and computes gradients for the whole batch in one pass (vectorized).

        Assumptions:
        - mini_batch is a list of tuples (x, y) where x has shape (n_in, 1) and y has shape (n_out, 1)
        - We form X of shape (n_in, m) and Y of shape (n_out, m) where m = len(mini_batch)
        """
        if len(mini_batch) == 0:
            return

        # Stack input and output column vectors horizontally to form matrices
        X = np.hstack([x if x.ndim == 2 else x.reshape(-1, 1) for x, _ in mini_batch])
        Y = np.hstack([y if y.ndim == 2 else y.reshape(-1, 1) for _, y in mini_batch])

        nabla_b, nabla_w = self.backprop_matrix(X, Y)
        m = X.shape[1]

        # Update weights and biases using the averaged gradients over the mini-batch
        self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)

        return (nabla_b, nabla_w)

    def backprop_matrix(self, X, Y):
        """
        Fully vectorized backpropagation over a mini-batch.

        Pseudocode:
        1. INPUT: X (n_in x m), Y (n_out x m), weights W_l, biases b_l for layers l=1..L
        2. FORWARD PASS:
             A_0 = X
             for l in 1..L:
                 Z_l = W_l @ A_{l-1} + b_l  # b_l broadcasts across the m columns
                 A_l = sigmoid(Z_l)
        3. BACKWARD PASS:
             delta_L = (A_L - Y) * sigmoid_prime(Z_L)            # (n_L x m)
             nabla_b_L = sum_columns(delta_L) -> shape (n_L x 1)
             nabla_w_L = delta_L @ A_{L-1}.T                     # (n_L x n_{L-1})
             for l in L-1 down to 1:
                 delta_l = (W_{l+1}.T @ delta_{l+1}) * sigmoid_prime(Z_l)
                 nabla_b_l = sum_columns(delta_l)
                 nabla_w_l = delta_l @ A_{l-1}.T
        4. RETURN lists [nabla_b_l], [nabla_w_l] for all layers.
        """
        # Containers for activations and z vectors (all matrix forms over the batch)
        activations = [X]  # A_0
        zs = []

        # Forward pass (matrix form)
        A = X
        for b, w in zip(self.biases, self.weights):
            Z = np.dot(w, A) + b  # b will broadcast across columns
            zs.append(Z)
            A = sigmoid(Z)
            activations.append(A)

        # Backward pass (matrix form)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # delta for the output layer
        delta = (activations[-1] - Y) * sigmoid_prime(zs[-1])  # shape (n_L, m)
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)     # sum across batch -> shape (n_L,1)
        nabla_w[-1] = np.dot(delta, activations[-2].T)         # shape (n_L, n_{L-1})

        # propagate backwards for remaining layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) 
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

        
    def cost_derivative(self, output_activations, y):
        """ Return the vector of partial derivatives \partial C_x
            \partial a for the output activations
        """
        return (output_activations - y)

# Helper functions
def sigmoid(z):
    return 1.0 / (1.0 +  np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function.""" 
    s = sigmoid(z)
    return s * (1 - s)


# net = Network([2, 3, 1])
# print(net.biases)  # I am expecting a 3 x 1 array + a 1 x 1 array
# print(net.weights) # I am expecting 3 X 2 matrx + 1 X 3 matrix


# Understanding the weight matrix
# A good way to help us design the weight matrix, is to think
# that we need to take this weight matrix and multiply that all the neurons in layer
# that it's from, that neurons we connect from will determine the number of rows, and the layer we are connecting to will 
# determine the number of columns. For example, if we have a layer of 2 connecting to a layer of 3 neurons, then the matrix of 
# connection would be a 3) x 2 matrix (as in 3 columns x 2 rows). Recall that matrix are represented as col x row

# Understanding the bias matrix:
# The bias will always have a row for every layer in the network besides the input layer, that means we just need the 
# number of neurons in layer and make that as our column size.
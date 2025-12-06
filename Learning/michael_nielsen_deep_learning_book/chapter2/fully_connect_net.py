import numpy as np
import pandas as pd
import random
import math


class Network:
    def __init__(self, sizes):
        # sizes is going to be a 1d array representing the number of neurons in each layer
        # say we take sizes as (2, 3, 1), then we are going to have 2 neurons at the start, 3 neurons in the hidden layer, and 1 neuron as the output layer
        self.num_layers = len(sizes)
        self.sizes = sizes
        # we know the biases only applies to the 1st - nth layer, excluding the 0th layer.
        self.biases = [np.random.randn(x, 1) for x in range(1, )]
        # we also know that weights are represened as matrices per layer,ex. from layer of neurons of 2 to layers of neurons of 3, would be a 3 x 2 matrix
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[0:-1], sizes[1:])]
    
    def feed_forward(self, A):
        # A is a layer of activation values
        for b, w in zip(self.biases, self.weights):
            A = sigmoid(w @ A + b)
        return A

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        '''
        training_data comes in the form of (x, y), x and y are all vectors, and x is of the size (input layer neurons, 1), and 
        y is of the size (output layer neurons, 1)
        '''

        for i in range(epochs):
            random.shuffle(mini_batches)
            # for every tranining cycle, we are using all of the training data, but we are passing those data in smaller batches known as mini-batch
            mini_batches = [training_data[i: i+ mini_batch_size] for i in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                print("mini_batch: ")
                print(mini_batch)
                self.update_mini_batch_matrix_based(mini_batch, eta)
            # now after updating the paramters using all the mini batches, we can perform some sort of evaluation process
            if test_data: # if the test data is available, then we can calculate metrics such as accuracy and etc.
                pass
    
    def update_mini_batch_matrix_based(self, mini_batch, eta):
        '''
        purpose of this function is to update the parameters using the mini_batches
        The idea of this funciton is really simple, we want to retrieve all the changes of the weights and 
        biases to minize the cost function for that mini batch, then we update the network's weights and biases in that desired direction
        '''
        if (len(mini_batch) == 0): return
        # We need two horizontally stacked matrices to form the X and Y matrices
        # X is the matrix of actual input values, and Y is the matrix of actual test values
        X = np.hstack([x if x.ndim == 2 else x.reshape(-1, 1) for x, _ in mini_batch])
        Y = np.hstack([y if y.ndim == 2 else y.reshape(-1, 1) for _, y in mini_batch])

        #nabla_b = []  # this represents all the changes to biases layer by layer
        #nabla_w = [] # this represents all the changes to weights layer by layer
        nabla_b, nabla_w = self.backprop_matrix_based(X, Y)

        m = len(mini_batch)

        self.weights = [w - (eta/m) *nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/m) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop_matrix_based(self, X, Y):
        '''
        X is essentially a matrx of all the input values from the mini batch
        Y is a matrix of all the true output values from the mini_batch used for computing the cost
        '''
        activations = []  # store all the activation values layer by layer, an array of matrix
        activations[0] = X
        zs = []

        # feedforward
        A = X
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, X) + b  # Taking the dot is designed to behave the same as matrix multiplication
            zs.append(z)
            A = sigmoid(z)
            activations.append(A)
        
        # Backprop
        # each element of b and w represents a layer-by-layer change to the entire biases/weights network. So each will be the same shape as weights and biases
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # compute the error terms
        delta = (activations[-1] - Y) * sigmoid_prime(zs[-1])  # delta is the change to the last layer
        nabla_b[-1] = np.sum(delta, axis =1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # propagate backwards
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = (self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = np.sum(delta, axis = 1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        return nabla_b, nabla_w

def sigmoid(z):
    return 1 / (1 + math.e**(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1- s)


        



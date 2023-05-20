import numpy as np
import numpy.matlib 
import math
import matplotlib.pyplot as plt
import csv

class NeuralNet:
    # WHOLE POINT IS TO MAKE NN GOOD AT ESTIMATING Q VALUES FOR A GIVEN STATE
    # This multi-layer perceptron will take a vector describing the current state of the baord
    # It will output a vector that contains each Q value for an action a on the current state
    def __init__(self,n_input,n_hidden,n_output, lr) -> None:
        # defining layers to nn
        # size of input layer
        self.n_input = n_input
        # input_layer will be vector describing current state of board (feature)
        # Helps extract more meaningful info from input feature
        self.n_hidden = n_hidden
        # n_output will be a vector containing the estimated Q values for each action on current state
        # number of neurons in output = number of actions possible for current state
        self.n_output = n_output

        # defining initial weights for nn (Using Xavier initialization)
        self.W1 = np.random.randn(n_hidden,n_input) * np.sqrt(1/(n_input))
        self.W2 = np.random.randn(n_output,n_hidden) * np.sqrt(1/(n_hidden))

        # defining biases for nn
        self.b_W1 = np.zeros((self.n_hidden,))
        self.b_W2 = np.zeros((self.n_output,))

        # Learning Rate
        self.lr = lr
    
    # returns a vector containing estimated q values for each action for the current state
    def forward(self, input_layer):
        # Activation: input layer -> hidden layer (weight * input) + bias
        h1 = np.dot(self.W1,input_layer) + self.b_W1
        # apply sigmoid activation function
        fh1 = 1/(1+np.exp(-h1))
        # Activation: hidden layer -> output layer (weight * h1) + bias
        h2 = np.dot(self.W2,fh1) + self.b_W2
        # apply sigmoid
        output = 1/(1+np.exp(-h2))
        
        # output contains the estmated Q values
        return output

    # gradient, update weights and biases of nn
    def backward(self, estimate, desired, input_layer):
        # Compute the error signal
        err = desired - estimate

        # Backpropagation: output -> hidden
        delta2 = estimate*(1-estimate) * err


        h1 = np.dot(self.W1,input_layer) + self.b_W1
        
        # apply sigmoid activation function
        fh1 = 1/(1+np.exp(-h1))

        # Compute gradient for W2 and b_W2
        dW2 = np.outer(delta2, fh1)
        db_W2 = delta2

        # Backpropagation: hidden -> input
        delta1 = fh1*(1-fh1) * np.dot(self.W2.T,delta2)
        dW1 = np.outer(delta1, input_layer)
        db_W1 = delta1

        self.W1 += self.lr * dW1
        self.W2 += self.lr * dW2

        self.b_W1 += self.lr * db_W1
        self.b_W2 += self.lr * db_W2


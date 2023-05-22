import numpy as np
import numpy.matlib 
import math
import matplotlib.pyplot as plt
import csv

class NeuralNet:
    '''
        Parameters
        ----------
        n_input: int
            number of neurons in input layer
        
        n_hidden: int
            number of neurons in hidden layer

        n_output: int
            number of neurons in output layer
    '''
    def __init__(self,n_input,n_hidden,n_output, lr) -> None:
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # defining initial weights for nn (Using Xavier initialization)
        self.W1 = np.random.randn(n_hidden,n_input) * np.sqrt(1/(n_input))
        self.W2 = np.random.randn(n_output,n_hidden) * np.sqrt(1/(n_hidden))

        # defining biases for nn
        self.b_W1 = np.zeros((self.n_hidden,))
        self.b_W2 = np.zeros((self.n_output,))

        # Learning Rate
        self.lr = lr
    
    '''
        Parameters
        ----------
        input-layer: np.array
            vector describing the current state of the board
        
        Returns
        -------
        output: np.array
            vector containing list of estimated Qvalues
    '''
    def forward(self, input_layer):
        # Activation: input layer -> hidden layer (weight * input) + bias
        h1 = np.dot(self.W1,input_layer) + self.b_W1
        # apply sigmoid activation function
        fh1 = 1/(1+np.exp(-h1))
        # Activation: hidden layer -> output layer (weight * h1) + bias
        h2 = np.dot(self.W2,fh1) + self.b_W2
        # apply sigmoid
        output = 1/(1+np.exp(-h2))
        
        return output

    
    '''
        Parameters
        ----------
        estimate: np.array
            Estimated Qvalues for current state <- output of forward
        
        desired: np.array
            desired Qvalues based on either Q-learning or SARSA

        input-layer: np.array
            vector describing the current state of the board
    '''
    def backward(self, estimate, desired, input_layer):
        # Compute the error signal
        err = desired - estimate
        # Backpropagation: output -> hidden
        delta2 = estimate*(1-estimate) * err

        # compute activation for input layer -> hidden layer
        h1 = np.dot(self.W1,input_layer) + self.b_W1
        # apply sigmoid activation function
        fh1 = 1/(1+np.exp(-h1))

        # Compute gradient for W2 and b_W2
        dW2 = np.outer(delta2, fh1)
        db_W2 = delta2

        # Backpropagation: hidden -> input
        delta1 = fh1*(1-fh1) * np.dot(self.W2.T,delta2)

        # Compute gradient for W2 and b_W2
        dW1 = np.outer(delta1, input_layer)
        db_W1 = delta1

        # update weights and biases
        self.W1 += self.lr * dW1
        self.W2 += self.lr * dW2

        self.b_W1 += self.lr * db_W1
        self.b_W2 += self.lr * db_W2



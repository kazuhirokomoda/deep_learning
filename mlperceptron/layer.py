# coding: utf-8
import numpy as np
import random

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1. - x)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1. - x * x

def ReLU(x):
    return x * (x > 0)

def d_ReLU(x):
    return 1. * (x > 0)


class Layer(object):
    """
    Layer class: represent each layer in multi layer perceptron
    """

    def __init__(self, input, n_in, n_out, W=None, activation=tanh):
        self.input = input
        self.n_in = n_in
        self.n_out = n_out

        if W is None:
            a = 1. / n_in
            W = np.random.uniform(low=-a, high=a, size=(self.n_out, self.n_in+1))
        self.W = W

        if activation == tanh:
            self.d_activation = d_tanh
        elif activation == sigmoid:
            self.d_activation = d_sigmoid
        elif activation == ReLU:
            self.d_activation = d_ReLU
        else:
            raise ValueError('activation function not supported.')
        self.activation = activation


    def forward(self, input=None, last=False):
        if input is not None:
            self.input = input
        linear_output = np.dot(self.W, self.input)
        self.output = self.activation(linear_output)
        if last != True:
            self.output = self.__add_bias(self.output)
        return self.output


    def backward(self, prev_layer_delta, lr=0.1):
        """
        prev: l+1
        next: l
        numpy.atleast_2d(*arys): View inputs as arrays with at least two dimensions.
        """
        if prev_layer_delta.shape[0] > 1:
            prev_layer_delta = prev_layer_delta[1::]
        delta_next = self.d_activation(self.input) * np.dot(self.W.T, prev_layer_delta)

        self.W -= lr * np.dot(np.atleast_2d(prev_layer_delta), np.atleast_2d(self.input).T)
        self.delta_next = delta_next
        return delta_next


    def __add_bias(self, x, axis=None):
        ones = np.ones(shape=(1, x.shape[1]))
        return np.concatenate((ones, x), axis=0)

    def get_outputlayer_delta(self, y):
        """
        Used only for the output layer in backward propagation
        """
        return (self.output - y) * self.d_activation(self.output)


# coding: utf-8
import numpy as np
import random

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer

from layer import *


class MyLabelBinarizer(LabelBinarizer):
    """
    1-of-K expression
    http://stackoverflow.com/questions/31947140/sklearn-labelbinarizer-returns-vector-when-there-are-2-classes
    """
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)


class MultiLayerPerceptron(object):
    """
    MultiLayerPerceptron class 
    """

    def __init__(self, n_in, n_out, hidden_layer_sizes=[]):
        """
        Keyword arguments:
        n_in -- dimension of input layer
        n_out -- dimension of output layer
        hidden_layer_sizes -- number of units in each hidden layer (e.g. [3, 3])
        """
        self.n_layers = len(hidden_layer_sizes)
        self.hidden_layers = []

        # add hidden layer(s)
        for i in range(self.n_layers):
            if i == 0:
                input_size = n_in
                layer_input = None
            else:
                input_size = hidden_layer_sizes[i-1]
                layer_input = None

            self.hidden_layers.append(
                Layer(layer_input, input_size, hidden_layer_sizes[i], activation=tanh)
            )

        # specify output layer
        self.output_layer = Layer(None, hidden_layer_sizes[self.n_layers-1], n_out, activation=sigmoid)


    def forward_mlp(self, input_data):
        calculated_data = input_data
        for i in range(self.n_layers):
            if i == 0:
                calculated_data = self.hidden_layers[i].forward(calculated_data)
            else:
                # add bias
                calculated_data = np.insert(calculated_data, 0, 1., axis=0)
                calculated_data = self.hidden_layers[i].forward(calculated_data)
        # add bias
        calculated_data = np.insert(calculated_data, 0, 1., axis=0)
        return self.output_layer.forward(input = calculated_data, last = True)


    def fit(self, X, t, lr=0.2, epochs=1000):
        """
        X: input
        t: label
        """
        # add bias to input
        ones = np.ones(shape=(X.shape[0], 1))
        X_with_bias = np.concatenate((ones, X), axis=1)

        self.label = t

        for i in range(epochs):
            #for n in range(0, X.shape[0]):
            index = np.random.randint(X_with_bias.shape[0])
            x = X_with_bias[index]

            # forward
            g = self.forward_mlp(x)

            # back propagation: renew weight
            output_delta = self.output_layer.get_outputlayer_delta(self.label[index])

            delta = self.output_layer.backward(output_delta, lr)

            for j in range(self.n_layers)[::-1]:
                # delta is not defined for the units whose output is always 1.
                delta_without_bias = delta[1::]
                delta = self.hidden_layers[j].backward(delta_without_bias, lr)

            self_predict = self.predict(X)
            loss_sum = np.sum((self_predict.T - self.label) ** 2)
            if i % 1000 == 0:
                print("epoch: {0:5d}, loss: {1:.5f}".format(i, loss_sum))


    def predict(self, X):
        """
        X.shape[0]: number of data
        """

        bias = np.ones(shape=(1, X.shape[0]))
        X_with_bias = np.concatenate((bias, X.T), axis=0)
        return self.forward_mlp(X_with_bias)



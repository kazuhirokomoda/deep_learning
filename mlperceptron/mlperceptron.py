# coding: utf-8
import numpy as np
import random

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer

from layer import *


class MyLabelBinarizer(LabelBinarizer):
    """
    TODO: 1-of-K expression
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
        hidden_layer_sizes -- number of units in each hidden layer (e.g. [3,3])
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
        self.output_layer = Layer(None, hidden_layer_sizes[self.n_layers-1], n_out, activation=tanh)


    def fit(self, X, t, lr=0.2, epochs=1000):
        """
        X: input
        t: label
        """

        self.label = t

        for i in range(epochs):
            #for n in range(0, X.shape[0]):
            index = np.random.randint(X.shape[0])
            x = X[index]

            g = np.atleast_2d(x).T

            # add bias to input
            ones = np.ones(shape=(1, g.shape[1]))
            g = np.concatenate((ones, g), axis=0)

            # forward
            for j in range(self.n_layers):
                g = self.hidden_layers[j].forward(g)
            g = self.output_layer.forward(input = g, last = True)

            # back propagation: renew weight
            output_delta = self.output_layer.get_outputlayer_delta(self.label[index])
            delta = self.output_layer.backward(output_delta, lr)
            for j in range(self.n_layers)[::-1]:
                delta = self.hidden_layers[j].backward(delta, lr)

            loss_sum = np.sum((self.predict(X) - self.label)**2)
            if i % 1000 == 0:
                print("epoch: {0:5d}, loss: {1:.5f}".format(i, loss_sum))


    def predict(self, X):
        """
        X.shape[0]: number of data
        """
        bias = np.ones(shape=(1, X.shape[0]))
        X = np.concatenate((bias, X.T), axis=0)

        for i in range(self.n_layers):
            X = self.hidden_layers[i].forward(X)
        return self.output_layer.forward(input = X, last = True)


if __name__ == "__main__":

    x_train = np.array([[0, 0], 
                        [0, 1],
                        [1, 0],
                        [1, 1]])
    y_train = np.array([0, 1, 1, 0])

    # TODO: 1-of-K expression: 1st column represents 1, 2nd column represents 0.
    #mlb = MyLabelBinarizer()
    #y_train_one_of_K = mlb.fit_transform(y_train)
    """
    y_train_one_of_K = np.array([[0, 1], 
                                 [1, 0],
                                 [1, 0], 
                                 [0, 1]])
    """

    # training
    mlp = MultiLayerPerceptron(len(x_train[0]), 1, [3]) # TODO: n_out=len(y_train_one_of_K[0])
    mlp.fit(x_train, y_train, epochs=10000) # TODO: t=y_train_one_of_K

    for x,y in zip(x_train, y_train):
        input = x
        #x = np.insert(x, 0, 1)
        #x = np.atleast_2d(x).T
        x = np.atleast_2d(x)
        print(input, y, mlp.predict(x))


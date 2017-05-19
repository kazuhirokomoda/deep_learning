# -*- coding: utf-8 -*-

"""
Multi Layer Perceptron test runner
"""

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from mlperceptron import *


def test_xor():
    """
    XOR
    """

    x_train = np.array([[0, 0], 
                        [0, 1],
                        [1, 0],
                        [1, 1]])
    y_train = np.array([0, 1, 1, 0])

    # 1-of-K expression: 1st column represents 1, 2nd column represents 0.
    mlb = MyLabelBinarizer()
    y_train_one_of_K = mlb.fit_transform(y_train)
    """
    y_train_one_of_K = np.array([[0, 1], 
                                 [1, 0],
                                 [1, 0], 
                                 [0, 1]])
    """

    # training
    mlp = MultiLayerPerceptron(len(x_train[0]), len(y_train_one_of_K[0]), [3])
    mlp.fit(x_train, y_train_one_of_K, epochs=10000)

    # evaluation (XOR)
    for x,y in zip(x_train, y_train):
        input = x
        x = np.atleast_2d(x)
        print(input, y, mlp.predict(x).T)


def test_digits():
    """
    MNIST digits
    """

    digits = load_digits()
    X = digits.data
    y = digits.target
    X /= X.max()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # 1-of-K expression
    # 0 => [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 1 => [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # ...
    # 9 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    lb = LabelBinarizer()
    y_train_one_of_K = lb.fit_transform(y_train)

    # training
    mlp = MultiLayerPerceptron(len(x_train[0]), len(y_train_one_of_K[0]), [500])
    mlp.fit(x_train, y_train_one_of_K, epochs=10000)

    # evaluation (MNIST digits)
    predictions = []
    for i in range(x_test.shape[0]):
        o = mlp.predict(np.atleast_2d(x_test[i]))
        # classify to class which has the largest output
        predictions.append(np.argmax(o))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))



if __name__ == "__main__":

    test_xor()
    test_digits()


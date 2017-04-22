# -*- coding: utf-8 -*-

"""
Perceptron test runner
"""

import numpy as np
from perceptron import Perceptron, phi, hyperplane, calculate_value, draw_graph

if __name__ == '__main__':

    # fix for making data points
    np.random.seed(0)

    # prepare data
    N = 100
    d = 2
    Xdata = np.random.randn(N, d)

    # ex. for d=2, array([[1],[2],[2]]) -> 1 + 2(x_1) + 2(x_2) = 0
    # [:, np.newaxis] to make vertical vector
    weight_true = np.random.randint(-5, 6, d+1)[:, np.newaxis]
    print(weight_true)

    # each element can either be 1 or -1
    label_true = np.array([1 if hyperplane(weight_true, x) > 0 else -1 for x in Xdata])[:, np.newaxis]

    # make Perceptron instance
    perceptron = Perceptron()

    # train weight
    perceptron.fit(Xdata, label_true)
    print(perceptron.w)

    # predict label from trained weight
    label_predict = perceptron.predict(Xdata)

    # evaluate
    num_correct_label = np.sum(label_predict == label_true)
    print('# of correctly predicted data:', num_correct_label)
    print('ratio: ', num_correct_label/N)

    # draw graph if possible
    draw_graph(Xdata, weight_true, label_true, perceptron)


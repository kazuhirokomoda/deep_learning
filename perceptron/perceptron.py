# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import random

# public symbols
__all__ = ['Perceptron']

class Perceptron(object):
    """
    Perceptron class 
    """
    def __init__(self):
        self.N = None
        self.d = None
        self.w = None
        self.rho = 0.5

    def fit(self, X, label):
        self.N = X.shape[0]
        self.d = X.shape[1]
        self.w = np.random.rand(self.d+1)[:, np.newaxis]

        # initialize
        np.random.seed()

        while True:
        #for i in range(100):
            data_index_list = list(range(self.N))
            random.shuffle(data_index_list)

            misses = 0
            for n in data_index_list:
                predict = calculate_value(self.w, X[n,:])
                label_value = label[n,:].sum()

                # perceptron learning rule
                if predict != label_value:
                    self.w += label_value * self.rho * phi(X[n,:])
                    misses += 1

            # learning finish when all data points are  correctly classified
            print(misses)
            if misses == 0:
            	break

    def predict(self, X):
        """
        predict label
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]

        label_predict = np.array([1 if hyperplane(self.w, x) > 0 else -1 for x in X])[:, np.newaxis]
        return label_predict


def phi(x):
	# vertical vector
    return np.concatenate(([1], x))[:, np.newaxis]

def hyperplane(w, x):
    return np.dot(w.T, phi(x))  # 真の分離平面 5x + 3y = 1

def calculate_value(w, X_n):
	# assumes shape (1, 1), so use sum()
    return np.sign(hyperplane(w, X_n).sum()).astype(np.int64)

def draw_graph(X, weight, label, perceptron):
    if perceptron.d == 2:
        xmax, xmin, ymax, ymin = 3, -3, 3, -3

        seq = np.arange(xmin, xmax, 0.02) # 0.02
        xlist, ylist = np.meshgrid(seq, seq)
        zlist = np.array([[calculate_value(weight, np.array([x_elem, y_elem])) for x_elem, y_elem in zip(x, y)] for x, y in zip(xlist, ylist)])

        # draw true separation hyperplane
        plt.pcolor(xlist, ylist, zlist, alpha=0.2, edgecolors='white')

        label_reshape = label.reshape(len(label))
        plt.plot(X[label_reshape== 1,0], X[label_reshape== 1,1], 'o', color='red')
        plt.plot(X[label_reshape== -1,0], X[label_reshape== -1,1], 'o', color='blue')

        # draw separation hyperplane based on the trained weight
        plain_x = np.linspace(xmin, xmax, 5)
        w_0, w_1, w_2 = perceptron.w[0], perceptron.w[1], perceptron.w[2]
        # (w_0)*(x_0) + (w_1)*(x_1) + (w_2)*(x_2) = 0 where x_0 = 1
        plain_y = - (w_1/w_2) * plain_x - (w_0/w_2)
        plt.plot(plain_x, plain_y, 'r-', color='black')

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.show()

""" implement a simple neural network with a sigmoid activation function """

import numpy as np


def sigmoid(x):
    # TODO: Implement sigmoid function
    return 1 / (1 + np.exp(-x))


def layer_1(inputs, weights, bias):
    """ repesents the simple layer that uses sigmoid activation function """
    h = (inputs * weights).sum() + bias
    return sigmoid(h)


inputs = np.array([0.7, -0.3])
weights = np.array([0.1, 0.8])
bias = -0.1

# TODO: Calculate the output
output = layer_1(inputs, weights, bias)

print('Output:')
print(output)

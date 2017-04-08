"""
use miniflow to calculate the Addition of two nodes
"""

# import Nodes classes
from miniflow import Input, Add, Mul, Linear, Sigmoid, MSE
# import helpler functions
from miniflow import topological_sort, forward_pass
import numpy as np


def calculate_add():
    # set Input nodes and the feed_dict
    x = Input()
    y = Input()
    z = Input()

    feed_dict = {x: 4, y: 10, z: 6}

    # create an Add node, which will also be the output node
    add = Add(x, y, z)
    mul = Mul(x, y, z)

    # to do (x + y) + y, add the following node
    # add2 = Add(add, y)

    # sort the nodes
    sorted_nodes = topological_sort(feed_dict)

    # do forward pass
    output = forward_pass(add, sorted_nodes)
    print(output)


def calculate_linear():
    # set Input nodes and the feed_dict
    inputs, weights, bias = Input(), Input(), Input()
    feed_dict = {
        inputs: [6, 14, 3],
        weights: [0.5, 0.25, 1.4],
        bias: 2
    }

    # create Linear node
    f = Linear(inputs, weights, bias)

    # sort the nodes
    graph = topological_sort(feed_dict)

    # do forward pass
    output = forward_pass(f, graph)
    print(output)


def calculate_linear_with_sigmoid():
    # set Input nodes and the feed_dict
    X, W, b = Input(), Input(), Input()
    X_ = np.array([[-1., -2.], [-1, -2]])
    W_ = np.array([[2., -3], [2., -3]])
    b_ = np.array([-3., -5])

    feed_dict = {X: X_, W: W_, b: b_}

    # create Linear node
    f = Linear(X, W, b)
    g = Sigmoid(f)

    # sort the node and do forward pass
    graph = topological_sort(feed_dict)
    output = forward_pass(g, graph)
    print(output)


def test_MSE():
    """ this part is just for testing the implementation of MSE """

    # set Inputs
    y, a = Input(), Input()

    y_ = np.array([1, 2, 3])
    a_ = np.array([4.5, 5, 10])
    feed_dict = {y: y_, a: a_}

    # set MSE node
    cost = MSE(y, a)

    # sort the node and calculate cost
    graph = topological_sort(feed_dict)
    forward_pass(cost, graph)
    print(cost.value)


if __name__ == '__main__':
    calculate_add()
    calculate_linear()
    calculate_linear_with_sigmoid()
    test_MSE()

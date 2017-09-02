"""
use miniflow to calculate the Addition of two nodes
"""

# import Nodes classes
from miniflow import Input, Add, Mul, Linear, Sigmoid, MSE
# import helpler functions
from miniflow import topological_sort, forward_pass, forward_and_backward, sgd_update
import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample


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


def test_forward_and_backward():
    X, W, b = Input(), Input(), Input()
    y = Input()
    f = Linear(X, W, b)
    a = Sigmoid(f)
    cost = MSE(y, a)

    X_ = np.array([[-1., -2.], [-1, -2]])
    W_ = np.array([[2.], [3.]])
    b_ = np.array([-3.])
    y_ = np.array([1, 2])

    feed_dict = {
        X: X_,
        y: y_,
        W: W_,
        b: b_,
    }

    graph = topological_sort(feed_dict)
    forward_and_backward(graph)
    # return the gradients for each Input
    gradients = [t.gradients[t] for t in [X, y, W, b]]

    """
    Expected output

    [array([[ -3.34017280e-05,  -5.01025919e-05],
           [ -6.68040138e-05,  -1.00206021e-04]]), array([[ 0.9999833],
           [ 1.9999833]]), array([[  5.01028709e-05],
           [  1.00205742e-04]]), array([ -5.01028709e-05])]
    """
    print(gradients)


def test_sgd_update():
    """
    using Boston housing price data to test sgd update and also the whole
    performance of miniflow
    """
    # load data
    data = load_boston()
    X_ = data['data']
    y_ = data['target']

    # normalize data
    X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

    n_features = X_.shape[1]
    n_hidden = 10
    W1_ = np.random.randn(n_features, n_hidden)
    b1_ = np.zeros(n_hidden)
    W2_ = np.random.randn(n_hidden, 1)
    b2_ = np.zeros(1)

    # Setup Neural Network
    X, y = Input(), Input()
    W1, W2, b1, b2 = Input(), Input(), Input(), Input()

    l1 = Linear(X, W1, b1)
    s = Sigmoid(l1)
    l2 = Linear(s, W2, b2)
    cost = MSE(y, l2)

    feed_dict = {
        X: X_,
        y: y_,
        W1: W1_,
        b1: b1_,
        W2: W2_,
        b2: b2_
    }

    # Set hypoparameters
    epochs = 1000

    # Set batch size
    m = X_.shape[0]
    batch_size = 11
    steps_per_epoch = m // batch_size

    graph = topological_sort(feed_dict)
    trainables = [W1, b1, W2, b2]

    print("Total number of examples = {}".format(m))

    for e in range(epochs):
        loss = 0
        for j in range(steps_per_epoch):
            # steps 1: select a batch of data randomly
            X_batch, y_batch = resample(X_, y_, n_samples=batch_size)
            X.value = X_batch
            y.value = y_batch

            # step 2: forward and backward propagation
            forward_and_backward(graph)

            # step 3: SGD update
            sgd_update(trainables)
            loss += graph[-1].value

            # step 4: repeat 1-3 for next epoch.

        print("Epoch: {}, Loss: {:.3f}".format(e + 1, loss / steps_per_epoch))


if __name__ == '__main__':
    calculate_add()
    calculate_linear()
    calculate_linear_with_sigmoid()
    test_MSE()
    test_forward_and_backward()
    test_sgd_update()

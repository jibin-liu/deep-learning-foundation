"""
use miniflow to calculate the Addition of two nodes
"""

from miniflow import Input, Add, Mul, Linear, topological_sort, forward_pass


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


if __name__ == '__main__':
    calculate_add()
    calculate_linear()

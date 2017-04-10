"""
Implement the miniflow architecture of neural network

By Jibin Liu
"""
from functools import reduce
import numpy as np


class Node(object):
    """
    this is the base class for nodes. The input, output, function nodes
    are inherited from this base class.
    """

    def __init__(self, inbound_nodes=[]):
        """
        receive the inbound nodes when initilize. Also set outbound_nodes to
        empty list, and initialize output variable
        """
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []
        self.value = None

        # the key in self.gradients is the inbound_nodes,
        # the value is the partials of this node with respect to that inbound_node.
        self.gradients = {}

        # add current node as outbound node for all of its inbound nodes
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    def forward(self):
        """
        forward propagation.

        use the value from inbound nodes to calculate the output value. Actual
        forward method should be implemented in subclasses.
        """
        raise NotImplementedError

    def backward(self):
        """
        backward propagation.

        Use the gradient from outbound_nodes, and the derivative of this node
        with respect to its inbound_node, to calculate the gradient loss for
        its inbound_nodes.
        """
        raise NotImplementedError


class Input(Node):
    """
    Input is the class for nodes that take input. They don't have the
    actual inbound_nodes
    """

    def __init__(self):
        super(Input, self).__init__()

    def forward(self, value=None):
        """
        Only for Input nodes, when calling forward method on them, they will pass
        the input directly to its output value.
        """
        if value:
            self.value = value

    def backward(self):
        """ Input node has not inputs, so the derivative is zero """
        self.gradients = {self: 0}

        # for weights and bias, we need to sum up the grad_cost from outbound_nodes
        # this can also be helpful to access the total grad_cost for Input nodes
        # see usage in sgd_update
        for node in self.outbound_nodes:
            grad_cost = node.gradients[self]
            self.gradients[self] += grad_cost


class Add(Node):
    """ The Add class is for nodes that do addition math. """

    def __init__(self, *args):
        super(Add, self).__init__(list(args))

    def forward(self):
        """ Add nodes will add the value of inbound_nodes """
        self.value = sum([n.value for n in self.inbound_nodes])

    def backward(self):
        """ for Add node, the derivative with respect to its inputs is 1 """
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        for node in self.outbound_nodes:
            grad_cost = node.gradients[self]

            for in_node in self.gradients:
                self.gradients[in_node] += grad_cost * 1


class Mul(Node):
    """ The Mul class is for nodes that do multiplication """

    def __init__(self, *args):
        super(Mul, self).__init__(list(args))

    def forward(self):
        """ Mul nodes will multiply the values of inbound_nodes """
        self.value = reduce(lambda x, y: x * y, [n.value for n in self.inbound_nodes])

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        for node in self.outbound_nodes:
            grad_cost = node.gradients[self]

            # the derivative of multiplication function with respect to an input
            # is the multiplication of all other input nodes.
            for in_node in self.inbound_nodes:
                self.gradients[in_node] += grad_cost * self.derivative(in_node)

    def derivative(self, node):
        """
        return the derivative of this Mul node with respect to the input node
        """
        other_nodes = [n.value for n in self.inbound_nodes if n is not node]
        return reduce(lambda x, y: x * y, other_nodes)


class Linear(Node):
    """ The Linear class is for linear combination """

    def __init__(self, x, w, b):
        super(Linear, self).__init__([x, w, b])

    def forward(self):
        inputs, weights, bias = [n.value for n in self.inbound_nodes]
        self.value = np.dot(inputs, weights) + bias

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        # sum grad_cost from each outbound_nodes to each input nodes.
        for node in self.outbound_nodes:
            grad_cost = node.gradients[self]
            inputs, weights, bias = self.inbound_nodes

            self.gradients[inputs] += np.dot(grad_cost, weights.value.T)
            self.gradients[weights] += np.dot(inputs.value.T, grad_cost)
            # not quite understand the following line, why need sum?
            self.gradients[bias] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Node):
    """ represents a node that used as a sigmoid activation gate """

    def __init__(self, node):
        super(Sigmoid, self).__init__([node])

    def forward(self):
        self.value = self._sigmoid_func(self.inbound_nodes[0].value)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        for node in self.outbound_nodes:
            grad_cost = node.gradients[self]
            # self._sigmoid has been calcuated during forward propagation
            inputs = self.inbound_nodes[0]
            self.gradients[inputs] += grad_cost * self._sigmoid_prime()

    def _sigmoid_func(self, x):
        self._sigmoid = 1. / (1. + np.exp(-x))  # 1. make sure it's float
        return self._sigmoid

    def _sigmoid_prime(self):
        return self._sigmoid * (1. - self._sigmoid)


class MSE(Node):
    """ represents a virtual node that calculates the cost/loss """

    def __init__(self, expected, calculated):
        super(MSE, self).__init__([expected, calculated])

    def forward(self):
        # the reshape is used to avoid possible broadcast error

        # for example, subtracting a matrice of (3, ) from (3, 1) will result
        # a shape of (3, 3), while what we want is actually (3, 1)

        # the calculated result "a" usually in shape (n, 1), while the expected
        # result is in shape (n, )

        expected, calculated = [n.value.reshape(-1, 1) for n in self.inbound_nodes]
        self.value = np.square(expected - calculated).sum() / len(expected)

        # save m and (y-a) for backward calculation
        self.m = len(expected)
        self.diff = expected - calculated

    def backward(self):
        # calculate derivative of MSE node with respect to calculated value
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff


# ---------- helpler functions -------------

def topological_sort(feed_dict):
    """
    Use topological sort to get the order for calculating each node.

    The input is a dictionary in which the key is the Input node, and the
    value is the Input node's value.
    """

    # first construct the graph
    nodes_to_add = list(feed_dict.keys())
    graph = dict()

    while nodes_to_add:
        node = nodes_to_add.pop(0)
        if node not in graph:
            graph[node] = {'in': set(), 'out': set()}

        for out_node in node.outbound_nodes:
            if out_node not in graph:
                graph[out_node] = {'in': set(), 'out': set()}

            graph[node]['out'].add(out_node)
            graph[out_node]['in'].add(node)
            nodes_to_add.append(out_node)

    # sort the graph and return ordered list
    ret_list = []
    s = set(feed_dict.keys())

    while s:
        node = s.pop()

        # if it's Input node, get its value from feed_dict
        if isinstance(node, Input):
            node.value = feed_dict[node]

        ret_list.append(node)

        for out_node in node.outbound_nodes:
            graph[node]['out'].remove(out_node)
            graph[out_node]['in'].remove(node)
            if not graph[out_node]['in']:
                s.add(out_node)

    return ret_list


def forward_pass(output_node, sorted_nodes):
    """
    do forward pass by calling forword method on all sorted_nodes, then
    return the value of output_node
    """

    for node in sorted_nodes:
        node.forward()

    return output_node.value


def forward_and_backward(sorted_nodes):
    """
    do a forward pass and a backward pass using the sorted list of nodes
    """
    for node in sorted_nodes:
        node.forward()

    for node in reversed(sorted_nodes):
        # another way to call reversed iteration on a list is [::-1]
        node.backward()


def sgd_update(trainables, learning_rate=1e-2):
    """
    update the trainables using grad_cost from their outbound_nodes.

    @trainables: a list of nodes whose value should be updated.
    @learning_rate: learning rate for neural network
    """
    for node in trainables:
        grad_cost = node.gradients[node]
        node.value -= learning_rate * grad_cost

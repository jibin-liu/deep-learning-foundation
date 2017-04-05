"""
Implement the miniflow architecture of neural network

By Jibin Liu
"""


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

        # add current node as outbound node for all of its inbound nodes
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    def forward(self):
        """
        forward propagation.

        use the value from inbound nodes to calculate the output value. Actual
        forward method should be implemented in subclasses.
        """
        raise NotImplemented


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


class Add(Node):
    """ The Add class is for nodes that do addition math. """

    def __init__(self, x, y):
        super(Add, self).__init__([x, y])

    def forward(self):
        """ Add nodes will add the value of inbound_nodes """
        self.value = sum([n.value for n in self.inbound_nodes])


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

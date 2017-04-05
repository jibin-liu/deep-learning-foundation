"""
use miniflow to calculate the Addition of two nodes
"""

from miniflow import Input, Add, Mul, topological_sort, forward_pass

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

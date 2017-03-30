import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])  # shape (, 3)
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])  # shape (3, 2)

weights_hidden_output = np.array([0.1, -0.3])  # shape (2, )

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)  # shape (, 2)  h
hidden_layer_output = sigmoid(hidden_layer_input)  # shape (, 2)  a

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)  # shape (,)  H
output = sigmoid(output_layer_in)  # shape (,)  y^

## Backwards pass
## TODO: Calculate error
error = target - output  # shape (,)  y - y^

# TODO: Calculate error gradient for output layer
del_err_output = error * output * (1 - output)  # shape (,)

# TODO: Calculate error gradient for hidden layer
del_err_hidden = np.dot(weights_hidden_output, del_err_output) *\
                 hidden_layer_output * (1 - hidden_layer_output)  # shape (3, 2)

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * del_err_output * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate * del_err_hidden * x[:, None]

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)

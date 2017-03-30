""" train neural network using university admission data """

import numpy as np
from data_prep import features, targets, features_test, targets_test


def sigmoid(x):
    """ define sigmoid activation function """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """ define derivative of sigmoid function """
    # this is just for understanding purpose.
    # when doing calculations, sigmoid will be pre-calculated, and then used
    # to calculate sigmoid_prime. This is faster than calling this func.
    return sigmoid(x) / (1 - sigmoid(x))


# set seeds
np.random.seed(42)

# get input data and expected results
n_records, n_features = features.shape   # shape (360, 6)
x = features.values  # shape (360, 6)
y = targets

# weights is initialized to be a normal distribution with mean = 0 and std = 1
# shape (6, )
weights = np.random.normal(scale=(1 / np.sqrt(n_features)), size=n_features)
last_loss = None  # to compare errors at each epoch

# nerual network hyperparameters
epochs = 1000
learn_rate = 0.5

for e in range(epochs):
    # initialize del_w. shape (6, )
    # not necessary if using matrix operations below
    # del_w = np.zeros(weights.shape)

    # calculate linear multiplication
    h = np.dot(x, weights)  # shape (360, )

    # calculate neural network output
    nn_output = sigmoid(h)  # shape (360, )

    # calculate error term
    error_term = (y - nn_output) * (nn_output / (1 - nn_output))  # shape (360,)

    # calculate del_w
    del_w = np.dot(error_term, x)  # shape (6, )

    # update weights, using MSE in this case
    weights += (learn_rate / n_records) * del_w  # shape (6, )

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss


# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))

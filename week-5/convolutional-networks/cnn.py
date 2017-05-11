from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('.', one_hot=True, reshape=False)

# parameters
learning_rate = 0.00001
epochs = 10
batch_size = 128

# network parameters
n_classes = 10  # MNIST datasets have 10 classes
dropout = 0.75  # dropout rate

# weights and biases
weights = {
    'wc1': tf.random(shape=(5, 5, 1, 128))
    ''
}


def conv_layer():
    pass


def maxpooling_layer():
    pass


def fully_connected_layer():
    pass


def classification_layer():
    pass


def convnets():
    # convolution layer + max pooling
    # 28 * 28 * 1 -> 14 * 14 * 32
    convnet = conv_layer()
    convnet = maxpooling_layer()

    # convolution layer + max pooling
    # 14 * 14 * 32 -> 7 * 7 * 64
    convnet = conv_layer()
    convnet = maxpooling_layer()

    # fully connected layer
    # 7 * 7 * 64 -> 1 * 1024
    convnet = fully_connected_layer()

    # classifier
    # 1 * 1024 -> 1 * 10
    convnet = classification_layer()

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

ckpt_path = os.path.join(os.getcwd(), 'checkpoints')

# read datasets
mnist = input_data.read_data_sets('.', one_hot=True, reshape=False)

# setting parameters
learning_rate = 0.01
epoches = 20
batch_size = 128
display_step = 1

n_input = 784  # MNIST image shape (28 * 28)
n_classes = 10  # labels (0-9 digits)
n_hidden_1 = 256  # units on hidden layer 1
n_hidden_2 = 128  # units on hidden layer 2

# set weights and bias
weights = {
    'hidden_layer_1': tf.Variable(tf.truncated_normal(shape=(n_input, n_hidden_1))),
    'hidden_layer_2': tf.Variable(tf.truncated_normal(shape=(n_hidden_1, n_hidden_2))),
    'output': tf.Variable(tf.truncated_normal(shape=(n_hidden_2, n_classes)))
}

biases = {
    'hidden_layer_1': tf.Variable(tf.zeros(shape=(n_hidden_1))),
    'hidden_layer_2': tf.Variable(tf.zeros(shape=(n_hidden_2))),
    'output': tf.Variable(tf.zeros(shape=(n_classes)))
}


# set network input
x = tf.placeholder(tf.float32, [None, 28, 28, 1])  # 1 stands for single channel
y = tf.placeholder(tf.float32, [None, n_classes])

x_flat = tf.reshape(x, [-1, n_input])  # flatten the input

# set network
hidden_layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer_1']), biases['hidden_layer_1'])
hidden_layer_1 = tf.nn.relu(hidden_layer_1)

hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, weights['hidden_layer_2']), biases['hidden_layer_2'])
hidden_layer_2 = tf.nn.relu(hidden_layer_2)

output = tf.add(tf.matmul(hidden_layer_2, weights['output']), biases['output'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,
                                                              labels=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


if __name__ == '__main__':
    # run network in session
    last_ckpt = tf.train.latest_checkpoint(ckpt_path)
    saver = tf.train.Saver()

    with tf.Session() as session:

        if not last_ckpt:
            print('Training Model.')
            session.run(tf.global_variables_initializer())

            for epoch in range(epoches):
                total_bathes = int(mnist.train.num_examples / batch_size)
                for batch in range(total_bathes):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    session.run(optimizer, feed_dict={x: batch_x, y: batch_y})

                # display progress
                if epoch % display_step == 0:
                    c = session.run(cost, feed_dict={x: batch_x, y: batch_y})
                    print('Epoch:', '{:4d}'.format(epoch + 1),
                          'cost=', '{:.9f}'.format(c))

            print('Optimization finished!')

            saver.save(session, os.path.join(ckpt_path, 'model'), global_step=epoches)
            print('Trained Model Saved.')

        else:
            # restore trained model
            # tf.reset_default_graph()  # if turned on, will have graph error
            saver.restore(session, last_ckpt)
            print('Trained Model Restored.')

            # test model
            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            print('Accuracy:', accuracy.eval({x: mnist.test.images,
                                              y: mnist.test.labels}))

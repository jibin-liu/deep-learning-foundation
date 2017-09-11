import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.animation as animation

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# build the model
inputs = tf.placeholder(tf.float32, [None, 784], name='inputs')
targets = tf.placeholder(tf.float32, [None, 10], name='targets')
W = tf.Variable(tf.truncated_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

logits = tf.matmul(inputs, W) + b
predictions = tf.nn.softmax(logits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    fig = plt.figure()
    plt.xlabel = 'Epoches'
    plt.ylabel = 'Accuracy'

    axes = plt.gca()
    axes.set_xlim(0, 1000)
    axes.set_ylim(0, 1)

    xdata = []
    ydata = []

    def animate(i):
        for epoch in range(1000):
            # get batch and train the model
            batch_x, batch_y = mnist.train.next_batch(100)
            train_feed = {inputs: batch_x, targets: batch_y}
            _ = sess.run(optimizer, feed_dict=train_feed)

            # calculate training accuracy and loss
            train_accu, train_loss = sess.run([accuracy, loss], feed_dict=train_feed)

            xdata.append(epoch)
            ydata.append(train_accu)
            axes.clear()
            axes.plot(xdata, ydata, 'r-')

            # calculate the validation accuracy and loss
            valid_x, valid_y = mnist.validation.images, mnist.validation.labels
            valid_feed = {inputs: valid_x, targets: valid_y}
            valid_accu, valid_loss = sess.run([accuracy, loss], feed_dict=valid_feed)

    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()

    # calculate the testing accuracy
    test_x, test_y = mnist.test.images, mnist.test.labels
    test_feed = {inputs: test_x, targets: test_y}
    test_accuracy = sess.run(accuracy, feed_dict=test_feed)

    print('Testing Accuracy={}'.format(test_accuracy))

import tensorflow as tf
import numpy as np
from setting import defaultPath
from tensorflow.examples.tutorials.mnist import  input_data
import os

current_dir = os.path.abspath(os.curdir)
current_dir = os.path.join(current_dir,"/tmp/mnist_logs")
print(current_dir)
mnist  = input_data.read_data_sets(defaultPath.Mnist_data_path,one_hot=True)
sess = tf.InteractiveSession()

# create the model
# name -> the graph representation
x = tf.placeholder(tf.float32, [None, 784], name = "x-input")
W = tf.Variable(tf.zeros([784, 10]), name = "weights")
b = tf.Variable(tf.zeros([10]), name = "bias")

# use a name_scope to organize nodes in the graph visualizer
# tf.name_scope -> the graph representation
with tf.name_scope("Wx_b") as scope:
    y = tf.nn.softmax(tf.matmul(x, W) + b)

# add summary ops to collect data
# historam summary
w_hist = tf.summary.histogram("weights", W)
b_hist = tf.summary.histogram("biases", b)
y_hist = tf.summary.histogram("y", y)

# define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10], name = "y-input")

# more name scopes will clean up the graph representation
with tf.name_scope("xent") as scoep:
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    ce_sum = tf.summary.scalar("cross entropy", cross_entropy)

with tf.name_scope("train") as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.name_scope("test") as scope:
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)

# merge all the summaries and write them out to /tmp/mnist_logs
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(os.path.join(current_dir,"/tmp/mnist_logs"), sess.graph_def)
tf.initialize_all_variables().run()

# train the model, and feed in test data and reord summaries every 10 steps
for i in range(100):
    if i % 10 == 0:
        feed = {x: mnist.test.images, y_: mnist.test.labels}
        result = sess.run([merged, accuracy], feed_dict = feed)
        summary_str = result[0]
        acc = result[1]
        writer.add_summary(summary_str, i)
        print(i, acc)
    else:
        batch_xs, batch_ys = mnist.train.next_batch(100)
        feed = {x: batch_xs, y_: batch_ys}
        sess.run(train_step, feed_dict = feed)
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
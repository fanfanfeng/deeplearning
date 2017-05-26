import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from tensorflow.examples.tutorials.mnist import  input_data
from setting import defaultPath
mnist = input_data.read_data_sets(defaultPath.Mnist_data_path,one_hot=True)

learning_rate = 0.001
traing_iters = 100000
batch_size = 128
display_step = 10

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder('float',[None,n_steps,n_input])
y = tf.placeholder("float",[None,n_classes])

weights = {
    'out':tf.Variable(tf.random_normal([2*n_hidden,n_classes]))
}

biases = {
    'out':tf.Variable(tf.random_normal([n_classes]))
}

def BiRNN(x,weights,biases):
    x = tf.unstack(x,n_steps,1)
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)

    outputs,_,_ = rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)

    return tf.matmul(outputs[-1],weights['out']) + biases['out']


pred = BiRNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step * batch_size < traing_iters:
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size,n_steps,n_input))
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})

            loss = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            print("Iter " + str(step *batch_size) +",Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ",Trainning Accuracy= "+ \
                  "{:.5f}".format(acc))
        step += 1

    print("Optimization Finished!")

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
    test_label = mnist.test.labels[:test_len]

    print("Testing  Accuracy:",sess.run(accuracy,feed_dict={x:test_data,y:test_label}))
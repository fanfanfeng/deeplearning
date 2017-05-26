import tensorflow as tf
import numbers as np
import matplotlib.pyplot as plt
from setting import defaultPath

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(defaultPath.Mnist_data_path,one_hot=True)

learning_rate = 0.01
training_epochs = 30
batch_size = 256
display_step = 1
examples_to_shoe = 10

n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784

x = tf.placeholder("float",[None,n_input])

weights = {
    'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'decoder_h1':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
    'decoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_input])),
}

biases = {
    'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2':tf.Variable(tf.random_normal([n_input])),
}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases["encoder_b2"]))

    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['decoder_h2']),biases['decoder_b2']))
    return layer_2


encoder_op = encoder(x)
decoder_op = decoder(encoder_op)


y_pred = decoder_op
y_true = x

cost = tf.reduce_mean(tf.pow(y_true - y_pred,2))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_xs})

        if epoch % display_step == 0 :
            print("Epoch:","%04d" % (epoch +1),"Cost = ","{:.9f}".format(c))

    print("Optimization finished!")

    encode_decode = sess.run(y_pred,feed_dict={x:mnist.test.images[:examples_to_shoe]})
    f,a = plt.subplots(2,10,figsize=(10,2))
    for i in range(examples_to_shoe):
        a[0][i].imshow(mnist.test.images[i].reshape((28,28)))
        a[1][i].imshow(encode_decode[i].reshape((28,28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()



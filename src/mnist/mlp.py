from tensorflow.examples.tutorials.mnist import input_data
from setting import defaultPath
mnist = input_data.read_data_sets(defaultPath.Mnist_data_path,one_hot=True)

import tensorflow as tf

learning_rate = 0.001
training_epochs = 100
batch_size = 100
display_step = 10


n_hidden_1 = 256
n_hidden_2 = 256
n_hidden_3 = 256

n_input = 784
n_classes = 10


x = tf.placeholder('float',[None,n_input])
y = tf.placeholder("float",[None,n_classes])

weights = {
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'h3':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
    'out':tf.Variable(tf.random_normal([n_hidden_3,n_classes]))
}

biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'b3':tf.Variable(tf.random_normal([n_hidden_3])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

def multilayer_preceptron(x,weights,biases):
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2,weights['h3']),biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    out_layer = tf.matmul(layer_3,weights['out']) + biases['out']
    return out_layer

pred = multilayer_preceptron(x,weights,biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoh in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})

            avg_cost += c / total_batch

        if epoh % display_step == 0:
            print("Epoch:","%04d" % (epoh + 1),"cost=","{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))


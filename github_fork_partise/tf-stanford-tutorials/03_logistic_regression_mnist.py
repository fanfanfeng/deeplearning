__author__ = 'fanfan'
from tensorflow.examples.tutorials import mnist
import tensorflow as tf
mnist_data_path = r"E:\github\deeplearning\deeplearning\data"
import time

learning_rate = 0.01
batch_size = 128
n_epochs = 10


mnist_data = mnist.input_data.read_data_sets(train_dir=mnist_data_path,one_hot=True)

X = tf.placeholder(tf.float32,shape=[batch_size,784],name="input")
Y = tf.placeholder(dtype=tf.int32,shape=[batch_size,10],name='output')


w = tf.Variable(tf.truncated_normal([784,10],stddev=0.1),name='weights')
b = tf.Variable(tf.random_normal([10]),name='bias')

logit = tf.matmul(X,w) + b

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=Y,name='loss')
loss = tf.reduce_mean(entropy)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist_data.train.num_examples/batch_size)
    for i in range(n_epochs):
        total_loss = 0
        for _ in range(n_batches):
            X_batch,Y_batch = mnist_data.train.next_batch(batch_size)
            _,loss_val = sess.run([train_op,loss],{X:X_batch,Y:Y_batch})
            total_loss += loss_val
        print("Average loss epoch {0}:{1}".format(i,total_loss/n_batches))
    print("Total time: {0} seconds".format(time.time() - start_time))
    print("Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(logit,1),tf.argmax(Y,1))
    accuray = tf.reduce_sum(tf.cast(correct_prediction,tf.int32))


    n_batches = int(mnist_data.test.num_examples/batch_size)
    total_correct_preds = 0

    for i in range(n_batches):
        X_batch,Y_batch = mnist_data.test.next_batch(batch_size)
        accuracy_batch = sess.run(accuray,feed_dict={X:X_batch,Y:Y_batch})
        total_correct_preds += accuracy_batch

    print("Accuracy {0}".format(total_correct_preds/mnist_data.test.num_examples))

import tensorflow as tf
from setting import defaultPath
from tensorflow.examples.tutorials.mnist import  input_data
import os

current_dir = os.path.abspath(os.curdir)
mnist  = input_data.read_data_sets(defaultPath.Mnist_data_path,one_hot=True)
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

logs_path = 'dashboard_log/basic/'

x = tf.placeholder(tf.float32,[None,784],name='InputData')
y = tf.placeholder(tf.float32,[None,10],name='LabelData')

W = tf.Variable(tf.zeros([784,10]),name='Weights')
b = tf.Variable(tf.zeros([10]),name='Bias')


with tf.name_scope("Model"):
    pred = tf.nn.softmax(tf.matmul(x,W) + b)

with tf.name_scope("Loss"):
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))

with tf.name_scope("SGD"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.name_scope("Accuracy"):
    acc = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(acc,tf.float32))


init = tf.global_variables_initializer()

tf.summary.scalar('loss',cost)
tf.summary.scalar('accuracy',acc)

merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            _,c,summary = sess.run([optimizer,cost,merged_summary_op],feed_dict={x:batch_xs,y:batch_ys})
            summary_writer.add_summary(summary,epoch * total_batch + i)
            avg_cost += c / total_batch

        if(epoch +1) % display_step == 0:
            print("Epoch:","%04d" % (epoch + 1),"Cost =","{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    print("Accuracy:",acc.eval({x:mnist.test.images,y:mnist.test.labels}))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=training:" + os.path.join(current_dir,logs_path) + \
          "\nThen open http://0.0.0.0:6006/ into you web browser")




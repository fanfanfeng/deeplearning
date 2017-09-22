__author__ = 'fanfan'
import os
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data_path = r"E:\github\deeplearning\deeplearning\data"

N_Class = 10
mnist = input_data.read_data_sets(mnist_data_path,one_hot=True)

LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 3

with tf.name_scope('data'):
    X = tf.placeholder(tf.float32,[None,784],name='x_placeholder')
    Y = tf.placeholder(tf.float32,[None,10],name='y_placeholder')

dropout = tf.placeholder(tf.float32,name="dropout")
global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')

if not os.path.exists('checkpoints/convnet_mnist'):
    os.mkdir('checkpoints/convnet_mnist')

with tf.variable_scope('conv1') as scope:
    images = tf.reshape(X,shape=[-1,28,28,1])
    kenerl = tf.get_variable(name='kenerl',shape=[5,5,1,32],initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable(name='b1',shape=[32],initializer=tf.random_normal_initializer())
    conv = tf.nn.conv2d(images,kenerl,strides=[1,1,1,1],padding="SAME")
    conv1 = tf.nn.relu(conv + b1,name=scope.name)

with tf.variable_scope("pool1") as scope:
    pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

with tf.variable_scope('conv2') as scope:
    kenerl = tf.get_variable(name='kenerls',shape=[5,5,32,64],initializer=tf.truncated_normal_initializer())
    b2 = tf.get_variable(name='b2',shape=[64],initializer=tf.random_normal_initializer())
    conv = tf.nn.conv2d(pool1,kenerl,strides=[1,1,1,1],padding="SAME")
    conv2 = tf.nn.relu(conv + b2,name=scope.name)

with tf.variable_scope("pool2") as scope:
    pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

with tf.variable_scope('fc') as scope:
    input_feature = 7*7*64

    fc_weights = tf.get_variable(name='fc_weights',shape=[7*7*64,1024],dtype=tf.float32)
    fc_biases = tf.get_variable(name='fc_biaes',shape=[1024],dtype=tf.float32)

    pool2 = tf.reshape(pool2,[-1,input_feature])
    fc = tf.nn.relu(tf.matmul(pool2,fc_weights) + fc_biases,name='relu')

    fc = tf.nn.dropout(fc,dropout,name='relu_dropout')

with tf.variable_scope('softemax_linear') as scope:
    softmax_weights = tf.get_variable('softmax_weights',shape=[1024,N_Class],dtype=tf.float32,initializer=tf.truncated_normal_initializer())
    softmax_biases = tf.get_variable('softmax_biases',shape=[N_Class],dtype=tf.float32,initializer=tf.random_normal_initializer())
    logits = tf.matmul(fc,softmax_weights) + softmax_biases

with tf.name_scope('loss'):
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=logits)
    loss = tf.reduce_mean(entropy,name='loss')

with tf.name_scope('summaries'):
    tf.summary.scalar('loss',loss)
    tf.summary.histogram('histogram loss',loss)
    summary_op = tf.summary.merge_all()

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss,global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./my_graph/mnist',sess.graph)
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)

    initial_step = global_step.eval()
    start_time = time.time()
    n_batches = int(mnist.train.num_examples/BATCH_SIZE)
    total_loss = 0.0
    for index in range(initial_step,n_batches * N_EPOCHS):
        X_batch,Y_batch = mnist.train.next_batch(BATCH_SIZE)
        _,loss_batch = sess.run([optimizer,loss],feed_dict = {X:X_batch,Y:Y_batch,dropout:DROPOUT})
        total_loss += loss_batch

        if (index + 1) % SKIP_STEP == 0:
            print("Aveage loss at  step {}:{:5.1f}".format(index + 1,total_loss/SKIP_STEP))
            total_loss = 0.0
            saver.save(sess,'checkpoints/convnet_mnist/mnist-convnet',index)

    print("Optimization Finished!")
    print("Total time:{0} seconds".format(time.time() - start_time))

    n_batches = int(mnist.test.num_examples/BATCH_SIZE)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch,Y_batch = mnist.test.next_batch(BATCH_SIZE)
        _,loss_batch,logits_batch = sess.run([optimizer,loss,logits],feed_dict = {X:X_batch,Y:Y_batch,dropout:1.0})

        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds,1),tf.argmax(Y_batch,1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds,tf.float32))
        total_correct_preds += sess.run(accuracy)

    print("Accuracy {0}".format(total_correct_preds/mnist.test.num_examples))




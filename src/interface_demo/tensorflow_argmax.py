# create by fanfan on 2017/7/17 0017

import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    a  = np.arange(24).reshape(2,3,4)
    print(a)

    b = tf.arg_max(a,dimension=2)
    tf_b = sess.run(b)
    print(tf_b.shape)
    print(tf_b)


    c = tf.reduce_max(a,axis=2)
    tf_c = sess.run(c)
    print(tf_c.shape)
    print(tf_c)

    d= tf.reduce_mean(a)
    tf_d = sess.run(d)
    print(tf_d.shape)
    print(tf_d)
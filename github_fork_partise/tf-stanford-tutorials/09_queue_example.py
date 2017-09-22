__author__ = 'fanfan'
import tensorflow as tf
import numpy as np

N_SAMPLES = 1000
NUM_thread = 4

data = 10 * np.random.randn(N_SAMPLES,4) + 1
target = np.random.randint(0,2,size = N_SAMPLES)

queue = tf.FIFOQueue(capacity=50,dtypes=[tf.float32,tf.float32],shapes=[[4],[]])

enqueue_op = queue.enqueue_many([data,target])
data_sample,label_sample = queue.dequeue()

qr = tf.train.QueueRunner(queue,[enqueue_op] * NUM_thread)
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(sess,coord=coord,start=True)
    try:
        for step in range(100):
            if coord.should_stop():
                break
            data_batch,label_batch = sess.run([data_sample,label_sample])
            print(data_batch)
            print(label_batch)
    except Exception as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(enqueue_threads)


import tensorflow as tf
import numpy as np

from tensorflow.contrib import  rnn

choice=0

#随机生成一个shape是(2,10,8)的矩阵，分布是标准正态函数
X = np.random.randn(2,10,8)

print(X)
if choice:
    cell = rnn.LSTMCell(num_units= 64)
else:
    cell = rnn.BasicLSTMCell(num_units=64)


outputs,status = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=cell,
    cell_bw=cell,
    dtype=tf.float64,
    inputs=X
)

output_fw,output_bw = outputs
status_fw,status_bw = status

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    result = sess.run({"output_fw": output_fw, "output_bw": output_bw, "status_fw": status_fw, "status_bw": status_bw},feed_dict=None)


    print(result["output_fw"].shape)
    print(result["output_bw"].shape)
    print(result["status_fw"].h.shape)
    print(result["status_bw"].h.shape)
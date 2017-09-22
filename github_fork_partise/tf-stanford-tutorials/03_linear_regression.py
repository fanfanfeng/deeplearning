__author__ = 'fanfan'
import numpy as np
import tensorflow as tf
import xlrd
import matplotlib.pyplot as plt

data_file_path = 'data/fire_theft.xls'

book = xlrd.open_workbook(data_file_path,encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data  = np.asarray([sheet.row_values(i) for i in range(1,sheet.nrows)])

n_samples = sheet.nrows - 1

X = tf.placeholder(tf.float32,name="fire")
Y = tf.placeholder(tf.float32,name="theft")

w = tf.Variable(0.0,name='weights')
b = tf.Variable(0.0,name='bias')

Y_predicted = X * w + b


loss = tf.square(Y - Y_predicted,name='loss')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/linear_reg',sess.graph)
    for i in range(1000):
        total_loss = 0
        for x,y in data:
            _,l = sess.run([optimizer,loss],feed_dict={X:x,Y:y})
            total_loss += l
        print("Epoch {0}:{1}".format(i,total_loss/n_samples))

    writer.close()

    w,b = sess.run([w,b])

X,Y = data.T[0],data.T[1]
plt.plot(X,Y,'bo',label='Real data')
plt.plot(X,X*w + b,'r',label="Predicted data")
plt.legend()
plt.show()

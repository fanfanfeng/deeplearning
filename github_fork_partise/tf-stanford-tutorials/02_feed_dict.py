__author__ = 'fanfan'

import tensorflow as tf

# create a placeholder of type float 32-bit, value is a vector of 3 elements
a = tf.placeholder(tf.float32,shape=[3])

# create a constant of type float 32-bit, value is a vector of 3 elements
b = tf.constant([5,5,5],dtype=tf.float32)

# use the placeholder as you would a constant
c = a + b  # short for tf.add(a, b)

with tf.Session() as sess:
    try:
        print(sess.run(c))
    except:
        print("InvalidArgumentError because a doesn’t have any value")
    finally:
        print(sess.run(c,{a:[1,2,3]})) # ouput  [6. 7. 8.]


a = tf.add(2,5)
b = tf.multiply(a,3)

with tf.Session() as sess:
    # define a dictionary that says to replace the value of 'a' with 15
    replace_dict = {a:15}

    # Run the session, passing in 'replace_dict' as the value to 'feed_dict'
    print(sess.run(b,feed_dict = replace_dict)) #ouput 45


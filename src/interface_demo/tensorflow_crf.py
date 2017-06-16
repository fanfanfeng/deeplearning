__author__ = 'fanfan'
import numpy as np
import  tensorflow as tf
from tensorflow.contrib import crf

# data setting
num_examples = 10
num_words = 20
num_features = 100
num_tags = 5

#random features
x = np.random.rand(num_examples,num_words,num_features).astype(np.float32)
print(x.shape)

#randomg tag indices representing the gold sequence
y = np.random.randint(num_tags,size=[num_examples,num_words]).astype(np.int32)
print(y.shape)
print(y)

#the length of the sequences
sequnce_lengths = np.full(num_examples,num_words-1,dtype=np.int32)
print(sequnce_lengths.shape)
print(sequnce_lengths)

with tf.Graph().as_default():
    with tf.Session() as session:
        x_t = tf.constant(x)
        y_t = tf.constant(y)
        sequnce_lengths_t = tf.constant(sequnce_lengths)

        #compute unary scores from a linear layer
        weights = tf.get_variable('weights',[num_features,num_tags])
        matricized_x_t = tf.reshape(x_t,[-1,num_features])
        matricized_unary_scores = tf.matmul(matricized_x_t,weights)

        unary_scores = tf.reshape(matricized_unary_scores,[num_examples,num_words,num_tags])


        #Compute the log-likeihood ofthe sequences and keep hte transition params for infernece at the test time
        log_likelihood,transition_params = crf.crf_log_likelihood(unary_scores,y_t,sequnce_lengths_t)

        #add a training op to tune the parameters
        loss = tf.reduce_mean(-log_likelihood)
        tran_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

        session.run(tf.global_variables_initializer())
        for i in range(1000):
            tf_unary_scores,tf_transition_params,_ = session.run([unary_scores,transition_params,tran_op])

            if i % 100 == 0:
                correct_labels = 0
                total_labels = 0
                for tf_unary_scores_,y_,sequence_length_ in zip(tf_unary_scores,y,sequnce_lengths):

                    #remove padding from the scores and tag sequence
                    tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
                    y_ = y_[:sequence_length_]

                    viterbi_sequence,_ = crf.viterbi_decode(tf_unary_scores_,tf_transition_params)

                    #evaluate word-level accuracy
                    correct_labels += np.sum(np.equal(viterbi_sequence,y_))
                    total_labels += sequence_length_
                accuracy = 100.0 * correct_labels/float(total_labels)
                print("accuracy: %.2f %%" % accuracy)











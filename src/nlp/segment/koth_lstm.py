# create by fanfan on 2017/7/21 0021

import numpy as np
import tensorflow as tf
import os
from setting import nlp_segment

from src.nlp.segment import data_loader
from tensorflow.contrib import rnn
from tensorflow.contrib import crf

FLAGS = tf.app.flags.FLAGS
from tensorflow.contrib.layers import l2_regularizer
tf.app.flags.DEFINE_string('log_dir', "logs", 'The log  dir')

class Model():
    def __init__(self):
        self.embeddingSize = nlp_segment.flags.embedding_size
        self.num_tags = nlp_segment.flags.num_tags
        self.num_hidden = nlp_segment.flags.num_hidden
        self.learning_rate = nlp_segment.flags.learning_rate
        self.batch_size = nlp_segment.flags.batch_size
        self.model_save_path = nlp_segment.model_save_path

        self.input = tf.placeholder(tf.int32,
                                  shape=[None, FLAGS.max_sentence_len],
                                  name="input_placeholder")

        self.label = tf.placeholder(tf.int32,
                                    shape=[None, FLAGS.max_sentence_len],
                                    name="label_placeholder")
        self.dropout = tf.placeholder(tf.float32,name="dropout")

        with tf.name_scope("embedding_layer"):
            self.word_embedding = tf.Variable(data_loader.load_w2v(nlp_segment.word_vec_path), name="word_embedding")
            inputs_embed = tf.nn.embedding_lookup(self.word_embedding,self.input)
            length = self.length(self.input)
            self.length_64 = tf.cast(length, tf.int64)
            reuse = None #if self.trainMode else True


            # if trainMode:
            #  word_vectors = tf.nn.dropout(word_vectors, 0.5)
            with tf.name_scope("rnn_fwbw") as scope:
                lstm_fw = rnn.LSTMCell(self.num_hidden,use_peepholes=True)
                lstm_bw = rnn.LSTMCell(self.num_hidden,use_peepholes=True)

                inputs = tf.unstack(inputs_embed, nlp_segment.flags.max_sentence_len, 1)
                outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw, lstm_bw, inputs, sequence_length=self.length_64,
                                                            dtype=tf.float32)
            output = tf.reshape(outputs, [-1, self.num_hidden * 2])
            #if self.trainMode:
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope('Softmax') as scope:
            self.W = tf.get_variable(shape=[self.num_hidden * 2, self.num_tags],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     name='weights',
                                     regularizer=l2_regularizer(0.001))
            self.b = tf.Variable(tf.zeros([self.num_tags], name='bias'))
            matricized_unary_scores = tf.matmul(output, self.W) + self.b
            # matricized_unary_scores = tf.nn.log_softmax(matricized_unary_scores)
            self.unary_scores = tf.reshape(
                matricized_unary_scores,
                [-1, FLAGS.max_sentence_len, self.num_tags])
        with tf.name_scope("crf"):
            self.transition_params = tf.get_variable(
                "transitions",
                shape=[self.num_tags, self.num_tags],
                initializer=self.initializer)
            log_likelihood, self.transition_params = crf.crf_log_likelihood(self.unary_scores, self.label, self.length_64,self.transition_params)
        self.loss = tf.reduce_mean(-log_likelihood)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()


    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def create_feed_dict(self,inputx,label,is_training):
        feed_dict = {
            self.input:inputx,
            self.dropout:1.0
        }
        if is_training:
            feed_dict[self.dropout] = nlp_segment.flags.dropout
            feed_dict[self.label] = label
        return feed_dict


    def train_step(self,sess,inputx,label,is_training):
        feed_dict = self.create_feed_dict(inputx, label,is_training)
        if is_training:
            fetch = [self.loss,self.train_op,self.transition_params]
            loss,_,transition = sess.run(fetch,feed_dict)
            return loss,transition

    def test_evaluate(self,sess, test_data,trainsMatrix):
        totalLen = test_data.sentence_words_id.shape[0]
        numBatch = int((totalLen- 1) / self.batch_size) + 1
        correct_labels = 0
        total_labels = 0
        for i in range(numBatch):
            endOff = (i + 1) * self.batch_size
            if endOff > totalLen:
                endOff = totalLen
            y = test_data.sentence_tags_id[i * self.batch_size:endOff]
            feed_dict = {self.input: test_data.sentence_words_id[i * self.batch_size:endOff],
                         self.dropout:1.0}
            unary_score_val, test_sequence_length_val = sess.run([self.unary_scores, self.length_64], feed_dict)
            for tf_unary_scores_, y_, sequence_length_ in zip(
                unary_score_val, y, test_sequence_length_val):
                # print("seg len:%d" % (sequence_length_))
                if sequence_length_ == 0:
                    continue
                tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
                y_ = y_[:sequence_length_]
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                    tf_unary_scores_, trainsMatrix)
                # Evaluate word-level accuracy.
                correct_labels += np.sum(np.equal(viterbi_sequence, y_))
                total_labels += sequence_length_
        accuracy = 100.0 * correct_labels / float(total_labels)
        print("Accuracy: %.3f%%" % accuracy)

    def restore_model(self,sess):
        check_point = tf.train.get_checkpoint_state(self.model_save_path)
        if check_point:
            self.saver.restore(sess,check_point.model_checkpoint_path)
        else:
            raise FileNotFoundError("not found the save model")




def main():
    graph = tf.Graph()
    with graph.as_default(),tf.Session() as sess:
        model = Model()

        train_manager = data_loader.BatchManager(nlp_segment.train_path,batch_size=nlp_segment.flags.batch_size)
        test_manager = data_loader.BatchManager(nlp_segment.test_path,batch_size=nlp_segment.flags.batch_size)
        init = tf.global_variables_initializer()
        sess.run(init)

        step = 0
        for _ in range(nlp_segment.flags.train_steps):
            for batch in train_manager.training_iter():
                step += 1
                inputs = batch['sentence_words_id']
                labels = batch['sentence_tags_id']
                loss,trainsMatrix = model.train_step(sess,inputs,labels,is_training=True)
                if (step + 1) % 100 == 0:
                    print("[%d] loss: [%r]" %
                          (step + 1, loss))
                    print(trainsMatrix)

                if (step + 1) % 1000 == 0:
                    model.test_evaluate(sess,test_manager,trainsMatrix)
                    model.saver.save(sess,model.model_save_path,step)

if __name__ == '__main__':
    main()



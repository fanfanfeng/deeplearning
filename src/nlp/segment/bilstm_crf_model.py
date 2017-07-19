__author__ = 'fanfan'
import tensorflow as tf
import numpy as np
from src.nlp.segment import data_loader
from setting import nlp_segment

from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib import rnn
from tensorflow.contrib import crf


class Model():
    def __init__(self):
        self.embedding_size = nlp_segment.flags.embedding_size
        self.num_tags = nlp_segment.flags.num_tags
        self.num_hidden = nlp_segment.flags.num_hidden
        self.learning_rate = nlp_segment.flags.learning_rate
        self.sentence_length = nlp_segment.flags.max_sentence_len
        self.word2vec_path = nlp_segment.word_vec_path

        self.model_save_path = nlp_segment.model_save_path
        self.hidden_layer_num = 1
        self.max_grad_norm = nlp_segment.flags.max_grad_norm

        self.input_x = tf.placeholder(dtype=tf.int32,shape=[None,self.sentence_length],name="input_x")
        self.labels = tf.placeholder(dtype=tf.int32,shape=[None,self.sentence_length],name='label')
        self.lengths = tf.placeholder(dtype=tf.int32,shape=[None],name='lengths')
        self.dropout = tf.placeholder(dtype=tf.float32,name='dropout')

        with tf.name_scope("embedding_layer"):
            self.word_embedding = tf.Variable(data_loader.load_w2v(),name="word_embedding")
            inputs_embed = tf.nn.embedding_lookup(self.word_embedding,self.input_x)
            # 对数据进行一些处理,把形状为(batch_size, n_steps, n_input)的输入变成长度为n_steps的列表,
            # 而其中元素形状为(batch_size, n_input), 这样符合LSTM单元的输入格式
            inputs_embed = tf.unstack(inputs_embed, self.sentence_length, 1)

        features = self.bi_lstm_layer(inputs_embed)

        with tf.variable_scope('Softmax') as scope:
            self.W = tf.get_variable(shape=[self.num_hidden *2,self.num_tags],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     name='weights',
                                     regularizer= l2_regularizer(0.001))
            self.b = tf.Variable(tf.zeros([self.num_tags],name='bias'))

            scores = tf.matmul(features,self.W) + self.b
            self.scores = tf.reshape(scores, [-1, self.sentence_length, self.num_tags])

        with tf.name_scope("crf"):
            log_likelihood,self.trans_form = crf.crf_log_likelihood(self.scores,self.labels,self.lengths)

        with tf.name_scope("output"):
            self.loss = tf.reduce_mean(-1.0 * log_likelihood)

        self.global_step = tf.Variable(0,name="global_step",trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        t_vars = tf.trainable_variables()
        grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss,t_vars),self.max_grad_norm)
        self.trans_op = self.optimizer.apply_gradients(zip(grads,t_vars),self.global_step)
        self.saver = tf.train.Saver()


    def lstm_cell(self):
        cell = rnn.LSTMCell(self.num_hidden)
        cell = rnn.DropoutWrapper(cell,self.dropout)
        return cell

    def bi_lstm_layer(self,inputs):
        if self.hidden_layer_num >1:
            lstm_fw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.hidden_layer_num)])
            lstm_bw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.hidden_layer_num)])
        else:
            lstm_fw = self.lstm_cell()
            lstm_bw = self.lstm_cell()

        outpus,_,_ = rnn.static_bidirectional_rnn(lstm_fw,lstm_bw,inputs,sequence_length=self.lengths,dtype=tf.float32)
        features = tf.reshape(outpus,[-1,self.num_hidden *2])
        return  features

    def create_feed_dict(self,inputx,label,lengths,is_training):
        feed_dict = {
            self.input_x:inputx,
            self.lengths:lengths,
            self.dropout:1.0
        }
        if is_training:
            feed_dict[self.dropout] = nlp_segment.flags.dropout
            feed_dict[self.labels] = label
        return feed_dict


    def train_step(self,sess,inputx,label,lengths,is_training):
        feed_dict = self.create_feed_dict(inputx, label, lengths,is_training)
        if is_training:
            fetch = [self.global_step,self.loss,self.trans_op,self.scores,self.trans_form]
            global_step,loss,_,_,_ = sess.run(fetch,feed_dict)
            return global_step,loss
        else:
            accuracy  = self.test_accurate(sess,inputx,label,lengths)
            return accuracy


    def test_accurate(self,sess,inputx,label,lengths):
        feed_dict = self.create_feed_dict(inputx,label,lengths,is_training=False)
        fetch = [self.scores,self.trans_form]
        scores,trans_form = sess.run(fetch, feed_dict)
        correct_num = 0
        total_labels = 0
        for score_,length_,label_ in zip(scores,lengths,label):
            if length_ ==0:
                continue
            score = score_[:length_]
            path,_ = crf.viterbi_decode(score,trans_form)
            label_path = label_[:length_]
            correct_num += np.sum(np.equal(path,label_path))
            total_labels += length_

        accuracy = 100.0 * correct_num / float(total_labels)
        return  accuracy


    def restore_model(self,sess):
        check_point = tf.train.get_checkpoint_state(self.model_save_path)
        if check_point:
            self.saver.restore(sess,check_point.model_checkpoint_path)
        else:
            raise FileNotFoundError("not found the save model")



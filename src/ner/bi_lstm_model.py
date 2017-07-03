__author__ = 'fanfan'
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood,viterbi_decode
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import regularizers
class model():
    def __init__(self,name,id_to_tag,parameters,word2vec):
        self.params = parameters
        self.learning_rate = self.params.learning_rate
        self.global_step = tf.Variable(0,trainable=False)
        self.tags = [ tag for i,tag in id_to_tag.items()]
        self.tags_num = len(self.tags)
        self.initializer = tf.random_uniform_initializer(-1,1)


        self.inputs = tf.placeholder(dtype=tf.int32,shape=[None,self.params.sentence_max_len],name="Inputs")
        self.labels = tf.placeholder(dtype=tf.int32,shape=[None,self.params.sentence_max_len],name="labels")
        self.lengths = tf.placeholder(dtype=tf.int32,shape=[None],name="Lengths")

        if self.params.feature_dim:
            self.features = tf.placeholder(dtype=tf.float32,shape=[None,self.params.word_max_len,self.params.feature_dim])

        self.dropout = tf.placeholder(dtype=tf.float32,name="Dropout")

        embedding = tf.Variable(word2vec,"word_emb")
        inputs_embed = tf.nn.embedding_lookup(embedding,self.inputs)

        inputs_embed = tf.nn.dropout(inputs_embed,self.dropout)

        if self.params.feature_dim:
            self.inputs = tf.concat(2,[self.inputs,self.features])

        rnn_features = self.bi_lstm_layer(inputs_embed)

        with tf.variable_scope("layer",initializer=self.initializer):
            w1 = tf.get_variable('W1',[self.params.neaurl_hidden_dim*2,self.tags_num],regularizer=regularizers.l2_regularizer(0.001))
            b1 = tf.get_variable('b1',[self.tags_num])
            scores = tf.matmul(rnn_features,w1) + b1
            self.scores = tf.reshape(scores,[-1,self.params.sentence_max_len,self.tags_num])

        with tf.variable_scope("crf"):
            log_likehood,self.transiton_params = crf_log_likelihood(self.scores,self.labels,self.lengths)
            self.loss = tf.reduce_mean(-1.0 * log_likehood)

        #gradient
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_vars = self.optimizer.compute_gradients(self.loss)
        capped_grads_vars = [(tf.clip_by_value(g,-self.params.clip,self.params.clip),v) for g,v in grads_vars]
        self.train_op = self.optimizer.apply_gradients(capped_grads_vars,self.global_step)

    def bi_lstm_layer(self,inputs):
        with tf.variable_scope("BILSTM"):
            fw_cell = rnn.LSTMCell(self.params.neaurl_hidden_dim,use_peepholes=True,initializer=self.initializer)
            bw_cell = rnn.LSTMCell(self.params.neaurl_hidden_dim,use_peepholes=True,initializer=self.initializer)

            outputs,_ = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,inputs,dtype=tf.float32)
            lstm_features = tf.reshape(outputs, [-1, self.params.neaurl_hidden_dim * 2])
            return lstm_features
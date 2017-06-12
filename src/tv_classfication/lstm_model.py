__author__ = 'fanfan'
import tensorflow as tf
from tensorflow.contrib import rnn
class LSTM():
    def __init__(self,config):
        self.initial_learning_rate = config.initial_learning_rate
        self.min_learning_rate = config.min_learning_rate
        self.decay_step = config.decay_step
        self.decay_rate = config.decay_rate
        self.num_step = config.num_step
        self.num_classes = config.num_classes
        self.hidden_neural_size = config.hidden_neural_size
        self.vocabulary_size = config.vocabulary_size
        self.embedding_dim = config.embedding_dim
        self.hidden_layer_num = config.hidden_layer_num
        self.w2v = config.w2v
        self.input_x = tf.placeholder(tf.int32,[None,self.num_step],name="input_x")
        self.input_y = tf.placeholder(tf.int32,[None,self.num_classes],name="input_y")

        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

        with tf.device('/cpu:0'),tf.name_scope("embedding_layer"):
            W = tf.Variable(self.w2v,name="W")
            inputs = tf.nn.embedding_lookup(W,self.input_x)
            inputs = tf.nn.dropout(inputs,self.dropout_keep_prob,name='dropout')

        if self.hidden_layer_num >1:
            lstmCells = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.hidden_layer_num)])
        else:
            lstmCells = self.lstm_cell()

        outputs,states = tf.nn.dynamic_rnn(lstmCells,inputs,dtype=tf.float32)
        with tf.name_scope("mean_pooling_layer"):
            output = outputs[:,self.num_step-1,:]

        with tf.name_scope("softmax_layer"):
            softmax_w = tf.get_variable('softmax_w',[self.hidden_neural_size,self.num_classes],dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b',[self.num_classes],dtype=tf.float32)

            self.logits = tf.add(tf.matmul(output,softmax_w),softmax_b)

        with tf.name_scope("output"):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits + 1e-10,labels=self.input_y)
            self.loss = tf.reduce_mean(self.cross_entropy,name="loss")

            self.predition = tf.argmax(self.logits,1,name='prediction')
            corrrect_prediction = tf.equal(self.predition,tf.argmax(self.input_y,1))
            self.correct_num = tf.reduce_sum(tf.cast(corrrect_prediction,tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(corrrect_prediction,tf.float32),name="accuracy")

        self.global_step = tf.Variable(0,name="global_step",trainable=False)
        self.learning_rate = tf.maximum(tf.train.exponential_decay(self.initial_learning_rate,self.global_step,self.decay_step,self.decay_rate,staircase=True),
                                        self.min_learning_rate)
        tvars = tf.trainable_variables()
        grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss,tvars),config.max_grad_norm)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer.apply_gradients(zip(grads,tvars))

        self.train_op = optimizer.apply_gradients(zip(grads,tvars),global_step=self.global_step)
        #self.summary = tf.summary.merge_all()





    def lstm_cell(self):
        lstm_cell = rnn.LSTMCell(self.hidden_neural_size,forget_bias=1.0)
        lstm_cell = rnn.DropoutWrapper(lstm_cell,output_keep_prob=self.dropout_keep_prob)
        return lstm_cell
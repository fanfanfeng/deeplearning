# create by fanfan on 2017/7/3 0003
import tensorflow as tf
from setting import  tv_classfication
from gensim.models import Word2Vec
import numpy as np
from tensorflow.contrib import rnn



def change_gensim_mode2array():
    model_path = tv_classfication.word2vec_path
    word2vec_model = Word2Vec.load(model_path)
    array_list = []
    for i in word2vec_model.wv.index2word:
        array_list.append(word2vec_model.wv[i])

    w2v = np.array(array_list)
    return w2v

class Bi_lstm():
    def __init__(self,flags = tv_classfication.flags):
        self.initial_learning_rate = flags.initial_learning_rate
        self.min_learning_rate = flags.min_learning_rate
        self.decay_step = flags.decay_step
        self.decay_rate = flags.decay_rate
        self.sentence_length = flags.sentence_length
        self.sentence_classes = flags.sentence_classes
        self.hidden_neural_size = flags.hidden_neural_size
        #self.input_dim_size = flags.input_dim_size
        self.hidden_layer_num = flags.hidden_layer_num
        #self.w2v =
        self.train_model_save_path = tv_classfication.train_model_bi_lstm

        self.input_x = tf.placeholder(tf.int32,[None,self.sentence_length])
        self.input_y = tf.placeholder(tf.int32,[None,self.sentence_classes])

        self.dropout = tf.placeholder(tf.float32,name='dropout')

        with tf.name_scope("embedding_layer"):
            self.W = tf.Variable(change_gensim_mode2array(),name="W")
            inputs = tf.nn.embedding_lookup(self.W,self.input_x)
            inputs = tf.nn.dropout(inputs,self.dropout,name="drouout_input")

            # 对数据进行一些处理,把形状为(batch_size, n_steps, n_input)的输入变成长度为n_steps的列表,
            # 而其中元素形状为(batch_size, n_input), 这样符合LSTM单元的输入格式
            inputs = tf.unstack(inputs,self.sentence_length,1)

            rnn_features = self.bilstm_layer(inputs)

        with tf.name_scope('softmax_layer'):
            softmax_w = tf.get_variable('softmax_w',[2 * self.hidden_neural_size,self.sentence_classes],dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b',[self.sentence_classes],dtype=tf.float32)

            self.logits = tf.add(tf.matmul(rnn_features,softmax_w),softmax_b,name='logits')

        with tf.name_scope("output"):
            self.cross_entry = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.input_y)
            self.loss = tf.reduce_mean(self.cross_entry,name="loss")

            self.prediction = tf.argmax(self.logits,1,name="prediction")

            correction_prediction = tf.equal(self.prediction,tf.argmax(self.input_y,1))

            self.accuracy = tf.reduce_mean(tf.cast(correction_prediction,tf.float32),name="accuracy")

        self.global_step = tf.Variable(0,name='global_step',trainable=False)
        self.learning_rate = tf.maximum(tf.train.exponential_decay(
            self.initial_learning_rate,self.global_step,self.decay_step,self.decay_rate,staircase=True
        ),self.min_learning_rate)


        tvars = tf.trainable_variables()
        grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss,tvars),flags.max_grad_norm)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads,tvars),global_step=self.global_step)
        self.summary = tf.summary.merge_all()

        self.saver = tf.train.Saver(tf.global_variables())



    def lstm_fw(self):
        lstm_fw = rnn.LSTMCell(self.hidden_neural_size)
        lstm_fw = rnn.DropoutWrapper(lstm_fw, self.dropout)

        return lstm_fw

    def lstm_bw(self):
        lstm_bw = rnn.LSTMCell(self.hidden_neural_size)
        lstm_bw = rnn.DropoutWrapper(lstm_bw, self.dropout)
        return lstm_bw

    def bilstm_layer(self,inputs):

        if self.hidden_layer_num >1:
            lstm_fw = rnn.MultiRNNCell([self.lstm_fw() for _ in range(self.hidden_layer_num)])
            lstm_bw = rnn.MultiRNNCell([self.lstm_bw() for _ in range(self.hidden_layer_num)])
        else:
            lstm_fw = self.lstm_fw()
            lstm_bw = self.lstm_bw()


        #outputs,_ = tf.nn.(cell_fw=lstm_fw,cell_bw=lstm_bw,inputs=inputs,dtype=tf.float32)
        outputs,_,_ = rnn.static_bidirectional_rnn(lstm_fw,lstm_bw,inputs,dtype=tf.float32)
        #outputs = tf.concat(outputs, 2)
        output = outputs[-1]
        return output

    def train_step(self,sess,is_train,inputX,inputY):
        feed_dict = self.create_feed_dict(is_train,inputX,inputY)
        fetches = [self.train_op,self.global_step,self.learning_rate,self.loss,self.accuracy]
        _,global_step,learning_rate,loss,accuracy = sess.run(fetches,feed_dict)
        return global_step,learning_rate,loss,accuracy

    def predict(self,sess,inputX):
        feed_dict = {
            self.dropout:1.0,
            self.input_x:inputX
        }

        fetches = [self.logits,self.prediction]

        logit,predict = sess.run(fetches,feed_dict)
        print(logit)
        return predict

    def create_feed_dict(self,is_train,innputX,inputY):
        feed_dict = {
            self.input_x:innputX,
            self.input_y:inputY,
            self.dropout:1.0
        }
        if is_train:
            feed_dict[self.dropout] = tv_classfication.flags.dropout

        return  feed_dict

    def restore_model(self,sess):
        check_point = tf.train.get_checkpoint_state(self.train_model_save_path)
        if not check_point:
            raise FileNotFoundError("not found model")
        self.saver.restore(sess,check_point.model_checkpoint_path)












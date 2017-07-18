# create by fanfan on 2017/7/11 0011
import tensorflow as tf
from setting import ner_tv
from common import data_convert
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib import crf
import numpy as np
from tensorflow.contrib import legacy_seq2seq


class Model():
    def __init__(self):
        self.initial_learning_rate = ner_tv.flags.initial_learning_rate
        self.tags_num = ner_tv.flags.tags_num
        self.sentence_length = ner_tv.flags.sentence_length
        self.hidden_neural_size = ner_tv.flags.hidden_neural_size
        self.hidden_layer_num = ner_tv.flags.hidden_layer_num
        self.initializer = tf.random_uniform_initializer(-1, 1)
        self.model_save_path = ner_tv.training_model_bi_lstm
        self.batch_size = ner_tv.flags.batch_size

        self.id_to_tag = ner_tv.id_to_tag


        self.input_x = tf.placeholder(dtype=tf.int32,shape=[None,self.sentence_length],name="input_x")
        self.labels = tf.placeholder(dtype=tf.float32,shape=[None,self.sentence_length,self.tags_num],name='label')
        self.lengths = tf.placeholder(dtype=tf.int32,shape=[None],name='lengths')
        self.dropout = tf.placeholder(dtype=tf.float32,name='dropout')

        with tf.name_scope("embedding_layer"):
            self.word_embedding = tf.Variable(data_convert.load_word2Vec(ner_tv.word2vec_path),name="word_embedding")
            inputs_embed = tf.nn.embedding_lookup(self.word_embedding,self.input_x)

            # 对数据进行一些处理,把形状为(batch_size, n_steps, n_input)的输入变成长度为n_steps的列表,
            # 而其中元素形状为(batch_size, n_input), 这样符合LSTM单元的输入格式
            inputs_embed = tf.unstack(inputs_embed, self.sentence_length, 1)

            rnn_features = self.bi_lstm_layer(inputs_embed)

        with tf.variable_scope("lsmt_crf_layer",initializer=self.initializer):
            w1 = tf.get_variable("w1",[self.hidden_neural_size *2,self.tags_num],regularizer=regularizers.l2_regularizer(0.001))
            b1 = tf.get_variable('b1',[self.tags_num])

            scores = tf.matmul(rnn_features,w1) + b1

        with tf.variable_scope("pridict"):
            self.prediction = tf.reshape(scores, [-1, self.sentence_length, self.tags_num])
            if 1:
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction,labels=self.labels))
            else:
                cross_entropy = self.labels * tf.log(self.prediction)
                cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
                mask = tf.sign(tf.reduce_max(tf.abs(self.labels), reduction_indices=2))
                cross_entropy *= mask
                cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
                cross_entropy /= tf.cast(self.lengths, tf.float32)
                self.loss = tf.reduce_mean(cross_entropy)
            #cross_entropy = #self.labels * #tf.log(self.prediction)
            #cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=1)
            #cross_entropy /= tf.cast(self.lengths, tf.float32)
            #self.loss =  tf.reduce_mean(-1.0 * prediction)

        #self.loss =  tf.reduce_sum(loss) / self.batch_size
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.initial_learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), ner_tv.flags.max_grad_norm)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        self.saver = tf.train.Saver()



    def lstm_cell(self):
        cell = rnn.LSTMCell(self.hidden_neural_size,use_peepholes=True,initializer=self.initializer)
        cell = rnn.DropoutWrapper(cell,self.dropout)
        return cell

    def bi_lstm_layer(self,inputs):
        if self.hidden_layer_num >1:
            lstm_fw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.hidden_layer_num)])
            lstm_bw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.hidden_layer_num)])
        else:
            lstm_fw = self.lstm_cell()
            lstm_bw = self.lstm_cell()

        outpus,_,_ = rnn.static_bidirectional_rnn(lstm_fw,lstm_bw,inputs,dtype=tf.float32)
        features = tf.reshape(outpus,[-1,self.hidden_neural_size *2])
        return  features

    def restore_model(self,sess):
        check_point = tf.train.get_checkpoint_state(self.model_save_path)
        if check_point:
            self.saver.restore(sess,check_point.model_checkpoint_path)
        else:
            raise FileNotFoundError("not found the save model")

    def create_feed_dict(self,inputx,label,lengths,is_training):
        feed_dict = {
            self.input_x:inputx,
            self.lengths:lengths,
            self.dropout:1.0
        }
        if is_training:
            feed_dict[self.dropout] = ner_tv.flags.dropout
            feed_dict[self.labels] = label
        return feed_dict


    def train_step(self,sess,inputx,label,lengths,is_training):
        feed_dict = self.create_feed_dict(inputx, label, lengths,is_training)
        if is_training:
            fetch = [self.global_step,self.loss,self.train_op]
            global_step,loss,_ = sess.run(fetch,feed_dict)
            return global_step,loss
        else:
            accuracy  = self.test_accurate(sess,inputx,label,lengths)
            return accuracy



    def test_accurate(self,sess,inputx,label,lengths):
        feed_dict = self.create_feed_dict(inputx,label,lengths,is_training=False)
        fetch = self.prediction
        prediction = sess.run(fetch, feed_dict)
        correct_num = 0
        total_num = 0

        pred_2_D = np.argmax(prediction,2)
        label_2_d = np.argmax(label,2)

        for pred,length_,label_ in zip(pred_2_D,lengths,label_2_d):
            pred = pred[:length_]
            label_ = label_[:length_]
            temp = np.equal(pred,label_)
            correct_num += np.sum(temp)
            total_num +=length_
            print(pred)




        accuracy = 100.0 * correct_num / float(total_num)
        return  accuracy




    def predict(self,sess,input,length):
        feed_dict = self.create_feed_dict(input, None, length, is_training=False)
        fetch = self.prediction
        prediction = sess.run(fetch, feed_dict)
        pred_2_D = np.argmax(prediction, 2)
        for pre, length_ in zip(pred_2_D, length):
            path = pre[:length_]
        return path

    def out_put_sentences(self,path,origin_sentence):
        out_sentent = ""
        temp = []
        type = 0
        for item,token in zip(path,origin_sentence):
            if item == 0:
                out_sentent += token + " "
            elif item == 1 or item == 2:
                currentType = self.id_to_tag[item].replace("B-","")
                if type != currentType and len(temp):
                    out_sentent += "".join(temp) + " "
                    temp = []
                else:
                    temp.append(token)

                type = currentType

        if len(temp)!=0:
            out_sentent += "".join(temp)

        return out_sentent













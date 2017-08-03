from __future__ import absolute_import
from __future__ import  division
from __future__ import print_function


import numpy as np
from tensorflow.contrib.layers import l2_regularizer
import tensorflow as tf
import os

from tensorflow.contrib import crf
from tensorflow.contrib import rnn

from setting import ner_tv

train_steps = 500000





from gensim.models import  Word2Vec
def load_word2Vec(model_path):
    word2vec_path = model_path
    word2vec = Word2Vec.load(word2vec_path)
    array_list = []
    for i in word2vec.wv.index2word:
        array_list.append(word2vec.wv[i])

    return np.array(array_list)




class Model:
    def __init__(self):
        self.embeddingSize = ner_tv.flags.embedding_dim
        self.distinctTagNum = ner_tv.flags.tags_num
        self.numHidden = ner_tv.flags.hidden_neural_size
        self.c2v = load_word2Vec(ner_tv.word2vec_path)
        self.words = tf.Variable(self.c2v,name = 'words')
        self.sentence_length = ner_tv.flags.sentence_length
        self.initial_learning_rate = ner_tv.flags.initial_learning_rate

        with tf.variable_scope('Softmax') as scope:
            self.W = tf.get_variable(shape=[self.numHidden *2,self.distinctTagNum],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     name='weights',
                                     regularizer= l2_regularizer(0.001))
            self.b = tf.Variable(tf.zeros([self.distinctTagNum],name='bias'))
        self.trains_params = None
        self.inp = tf.placeholder(tf.int32,shape=[None,self.sentence_length],name='input_placeholder')

        self.model_save_path = ner_tv.training_model_bi_lstm
        self.saver = tf.train.Saver()

    def restore_model(self,sess):
        check_point = tf.train.get_checkpoint_state(self.model_save_path)
        if check_point:
            self.saver.restore(sess,check_point.model_checkpoint_path)
        else:
            raise FileNotFoundError("not found the save model")




    def length(self,data):
        used = tf.sign(tf.reduce_max(tf.abs(data),reduction_indices=2))
        length = tf.reduce_sum(used,reduction_indices=1)
        length = tf.cast(length,tf.int32)
        return length

    def inference(self,X,reuse = None,trainMode=True):
        word_verctors = tf.nn.embedding_lookup(self.words,X)
        length = self.length(word_verctors)
        length_64 = tf.cast(length,tf.int64)
        if trainMode:
            word_verctors = tf.nn.dropout(word_verctors,0.5)

        with tf.variable_scope('rnn_fwbw',reuse =reuse) as scope:
            lstm_fw = rnn.LSTMCell(self.numHidden)
            lsmt_bw = rnn.LSTMCell(self.numHidden)

            inputs = tf.unstack(word_verctors,self.sentence_length,1)
            output,_,_ = rnn.static_bidirectional_rnn(lstm_fw,lsmt_bw,inputs,sequence_length=length_64,dtype=tf.float32)
            output = tf.reshape(output,[-1,self.numHidden * 2])

        matricized_unary_scores = tf.matmul(output,self.W) + self.b
        unary_scores = tf.reshape(matricized_unary_scores,
                                  [-1,self.sentence_length,self.distinctTagNum])
        return unary_scores,length

    def loss(self,X,Y):
        P,sequence_length = self.inference(X)
        log_likelihood,self.trains_params = crf.crf_log_likelihood(P,Y,sequence_length)
        loss = tf.reduce_mean(-log_likelihood)
        return loss

    def test_unary_score(self):
        P,sequence_length  = self.inference(self.inp,reuse=True,trainMode=False)
        return P,sequence_length

    def train_op(self,total_loss):
        return tf.train.AdamOptimizer(self.initial_learning_rate).minimize(total_loss)

    def training_step(self):
        pass


def read_csv(batch_size,file_name):
    filename_queue = tf.train.string_input_producer([file_name])
    reader = tf.TextLineReader(skip_header_lines=0)
    key,value = reader.read(filename_queue)

    decoded =  tf.decode_csv(value,field_delim=' ',
                             record_defaults=[[0] for i in range(ner_tv.flags.sentence_length*2)])
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size*50,
                                  min_after_dequeue=batch_size)



def inputs(path):
    whole = read_csv(ner_tv.flags.batch_size,path)
    features = tf.transpose(tf.stack(whole[0:ner_tv.flags.sentence_length]))
    label = tf.transpose(tf.stack(whole[ner_tv.flags.sentence_length:]))
    return features,label


def do_load_data(path):
    x = []
    y = []
    fp = open(path,'r')
    for line in fp.readlines():
        line = line.rstrip()
        if not line:
            continue
        ss = line.split(" ")
        assert (len(ss) == (ner_tv.flags.sentence_length *2))
        lx = []
        ly = []
        for i in range(ner_tv.flags.sentence_length):
            lx.append(int(ss[i]))
            ly.append(int(ss[i + ner_tv.flags.sentence_length]))
        x.append(lx)
        y.append(ly)
    fp.close()
    return np.array(x),np.array(y)

def train(total_loss):
    return tf.train.AdamOptimizer(ner_tv.flags.initial_learning_rate).minimize(total_loss)

def test_evaluate(sess,unary_score,test_sequence_length,transMatrix,inp,tX,tY):
    totalEqual = 0
    batchSize = ner_tv.flags.batch_size
    totalLen = tX.shape[0]
    numBatch = int((tX.shape[0] -1)/batchSize) +1
    correct_labels = 0
    total_labels = 0
    for i in range(numBatch):
        endOff = (i +1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        y = tY[i * batchSize:endOff]

        feed_dict = {inp: tX[i* batchSize:endOff]}
        unary_score_val,test_sequence_length_val = sess.run([unary_score,test_sequence_length],feed_dict)
        for tf_unary_scores_,y_,sequence_length_ in zip(
            unary_score_val,y,test_sequence_length_val):
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
            y_ = y_[:sequence_length_]
            viterbi_sequence,_ = crf.viterbi_decode(tf_unary_scores_,transMatrix)

            correct_labels += np.sum(np.equal(viterbi_sequence,y_))
            total_labels += sequence_length_
    accuracy = 100.0 * correct_labels/ float(total_labels)
    print('Accuracy: %.2f%%' % accuracy)

def main(argv):
    trainDataPath = os.path.join(os.path.dirname(ner_tv.data_path),"train.txt")
    testDataPath = os.path.join(os.path.dirname(ner_tv.data_path), "test.txt")

    graph = tf.Graph()
    with graph.as_default():
        model = Model()
        print('train data path:',trainDataPath)
        X,Y = inputs(trainDataPath)
        #tX,tY = do_load_data(testDataPath)
        total_loss = model.loss(X,Y)
        train_op = train(total_loss)
        #test_unary_score,test_sequence_length = model.test_unary_score()

        #sv = tf.train.Supervisor(graph=graph,logdir= model.model_save_path)
        #with sv.managed_session(master="") as sess:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_steps = train_steps
            for step in range(training_steps):
                #if sv.should_stop():
                    #break
                try:
                    _,trainsMatrix = sess.run([train_op,model.trains_params])
                    print(step)
                    if step % 100 == 0:
                        print('[%d] loss: [%r]' % (step,sess.run(total_loss)))
                    if step % 1000 ==0:
                        pass
                        #test_evaluate(sess,test_unary_score,test_sequence_length,
                                      #trainsMatrix,model.inp,tX,tY)
                except KeyboardInterrupt as e:
                    #sv.saver.save(sess,ner_tv.training_model_bi_lstm+'/model',global_step= step +1)
                    raise  e
            #sv.saver.save(sess,ner_tv.training_model_bi_lstm +"/finnal-model")
            #tf.train.write_graph(sess.graph_def,'models/','graph.pb')

if __name__ == '__main__':
    #main()
    tf.app.run()

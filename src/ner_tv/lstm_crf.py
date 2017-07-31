from __future__ import absolute_import
from __future__ import  division
from __future__ import print_function

import  numpy as np
import tensorflow as tf


from tensorflow.contrib import learn

import os
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib import crf

from setting import nlp_segment
from tensorflow.contrib import rnn
from gensim.models import Word2Vec
from common import data_convert





class Model:
    def __init__(self,embeddingSize,distinctTagNum,c2vPath,numHidden):
        self.embeddingSize = embeddingSize
        self.distinctTagNum = distinctTagNum
        self.numHidden = numHidden
        self.c2v = self.c2v(c2vPath)
        self.words = tf.Variable(self.c2v,name = 'words')
        with tf.variable_scope('Softmax') as scope:
            self.W = tf.get_variable(shape=[numHidden *2,distinctTagNum],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     name='weights',
                                     regularizer= l2_regularizer(0.001))
            self.b = tf.Variable(tf.zeros([distinctTagNum],name='bias'))
        self.trains_params = None
        self.inp = tf.placeholder(tf.int32,shape=[None,nlp_segment.flags.max_sentence_len],name='input_placeholder')



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

            inputs = tf.unstack(word_verctors,nlp_segment.flags.max_sentence_len,1)
            output,_,_ = rnn.static_bidirectional_rnn(lstm_fw,lsmt_bw,inputs,sequence_length=length_64,dtype=tf.float32)
            output = tf.reshape(output,[-1,self.numHidden * 2])

        matricized_unary_scores = tf.matmul(output,self.W) + self.b
        unary_scores = tf.reshape(matricized_unary_scores,
                                  [-1,nlp_segment.flags.max_sentence_len,self.distinctTagNum])
        return unary_scores,length

    def loss(self,X,Y):
        P,sequence_length = self.inference(X)
        log_likelihood,self.trains_params = crf.crf_log_likelihood(P,Y,sequence_length)
        loss = tf.reduce_mean(-log_likelihood)
        return loss

    def test_unary_score(self):
        P,sequence_length  = self.inference(self.inp,reuse=True,trainMode=False)
        return P,sequence_length

def read_csv(batch_size,file_name):
    filename_queue = tf.train.string_input_producer([file_name])
    reader = tf.TextLineReader(skip_header_lines=0)
    key,value = reader.read(filename_queue)

    decoded =  tf.decode_csv(value,field_delim=' ',
                             record_defaults=[[0] for i in range(nlp_segment.flags.max_sentence_len*2)])
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size*50,
                                  min_after_dequeue=batch_size)



def inputs(path):
    whole = read_csv(nlp_segment.flags.batch_size,path)
    features = tf.transpose(tf.stack(whole[0:nlp_segment.flags.max_sentence_len]))
    label = tf.transpose(tf.stack(whole[nlp_segment.flags.max_sentence_len:]))
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
        assert (len(ss) == (nlp_segment.flags.max_sentence_len *2))
        lx = []
        ly = []
        for i in range(nlp_segment.flags.max_sentence_len):
            lx.append(int(ss[i]))
            ly.append(int(ss[i + nlp_segment.flags.max_sentence_len]))
        x.append(lx)
        y.append(ly)
    fp.close()
    return np.array(x),np.array(y)

def train(total_loss):
    return tf.train.AdamOptimizer(nlp_segment.flags.learning_rate).minimize(total_loss)

def test_evaluate(sess,unary_score,test_sequence_length,transMatrix,inp,tX,tY):
    totalEqual = 0
    batchSize = nlp_segment.flags.batch_size
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
    trainDataPath = os.path.join(nlp_segment.data_path,"old/train.txt")
    testDataPath = os.path.join(nlp_segment.data_path, "old/test.txt")

    graph = tf.Graph()
    with graph.as_default():
        model = Model(nlp_segment.flags.embedding_size,nlp_segment.flags.num_tags,nlp_segment.word_vec_path,
                      nlp_segment.flags.num_hidden)
        print('train data path:',trainDataPath)
        X,Y = inputs(trainDataPath)
        tX,tY = do_load_data(testDataPath)
        total_loss = model.loss(X,Y)
        train_op = train(total_loss)
        test_unary_score,test_sequence_length = model.test_unary_score()
        sv = tf.train.Supervisor(graph=graph,logdir=nlp_segment.model_save_path)
        with sv.managed_session(master="") as sess:
            training_steps = nlp_segment.flags.train_steps
            for step in range(training_steps):
                if sv.should_stop():
                    break
                try:
                    _,trainsMatrix = sess.run([train_op,model.trains_params])
                    if step % 100 == 0:
                        print('[%d] loss: [%r]' % (step,sess.run(total_loss)))
                    if step % 1000 ==0:
                        test_evaluate(sess,test_unary_score,test_sequence_length,
                                      trainsMatrix,model.inp,tX,tY)
                except KeyboardInterrupt as e:
                    sv.saver.save(sess,nlp_segment.model_save_path+'/model',global_step= step +1)
                    raise  e
            sv.saver.save(sess,nlp_segment.model_save_path +"/finnal-model")
            #tf.train.write_graph(sess.graph_def,'models/','graph.pb')

if __name__ == '__main__':
    #main()
    tf.app.run()

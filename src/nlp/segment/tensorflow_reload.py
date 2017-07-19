import tensorflow as tf

import os
import numpy as np
from tensorflow.python.platform import gfile
from gensim.models import Word2Vec
from tensorflow.contrib import crf

tf.app.flags.DEFINE_string('word2vec_path','vec-kcws.txt','the word2vec data path')
tf.app.flags.DEFINE_integer('max_sentence_len',80,"max num of tokens per query")
tf.app.flags.DEFINE_integer('embedding_size',50,'embedding size')


pbDir = 'models/seg_model.pbtxt'
saveDir = 'logs/'


test = [u'机',u'器',u'学',u'习']

def makeVec():
    result = []
    wordVec = Word2Vec.load_word2vec_format(tf.app.flags.FLAGS.word2vec_path, binary=False)
    nn = len(test)
    for i in range(nn):
        result.append(wordVec.vocab[test[i]].index)
    for j in range(nn, tf.app.flags.FLAGS.max_sentence_len):
        result.append(0)
    return result




def load_w2v(path):
    fp = open(path, "r")
    print('load data from:', path)
    line = fp.readline().strip()
    ss = line.split(" ")
    total = int(ss[0])
    dim = int(ss[1])
    assert (dim == (tf.app.flags.FLAGS.embedding_size))
    ws = []
    mv = [0 for i in range(dim)]
    ws.append([0 for i in range(dim)])
    for t in range(total):
        line = fp.readline().rstrip()
        ss = line.split(" ")
        assert (len(ss) == (dim + 1))
        vals = []
        for i in range(1, dim + 1):
            fv = float(ss[i])
            mv[i - 1] += fv
            vals.append(fv)
        ws.append(vals)

    for i in range(dim):
        mv[i] = mv[i] / total

    ws.append(mv)
    fp.close()
    return np.asarray(ws, dtype=np.float32)

testVerb = makeVec()
print(testVerb)


with tf.Session() as sess:
    print('load graph')
    with gfile.FastGFile(pbDir,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def,name="")



        trans_tensors = sess.graph.get_tensor_by_name('transitions:0')

        result = sess.run(trans_tensors)
        print(result)


        inp = sess.graph.get_tensor_by_name('input_placeholder:0')
        print(inp)
        #inp = tf.placeholder(tf.int32, shape=[None, 80], name='input_placeholder')

        npVerb = np.array(testVerb).reshape([1,80])
        #input_tensor = tf.Variable(npVerb,dtype=np.int32)
        #print(input_tensor)
        feed_dict = {inp:npVerb }
        for v in sess.graph.get_operations():
            print(v.name)
        scoreNode = sess.graph.get_tensor_by_name('Reshape_7:0')

        unary_score_val = sess.run(scoreNode,feed_dict)
        print(unary_score_val.shape)


        viterbi_sequence, _ = crf.viterbi_decode(unary_score_val[0], result)
        print(viterbi_sequence)

        fenchiResult = ""
        for i in range(len(test)):
            if viterbi_sequence[i] == 0:
                fenchiResult += test[i] +" "
            elif viterbi_sequence[i] == 1:
                fenchiResult += test[i]
            elif viterbi_sequence[i] == 2:
                fenchiResult += test[i]
            elif viterbi_sequence[i] == 3:
                fenchiResult += test[i] + " "
        print(fenchiResult)
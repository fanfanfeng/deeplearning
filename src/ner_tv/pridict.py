# create by fanfan on 2017/7/14 0014

from src.ner_tv import lstm_crf
import tensorflow as tf
from setting import ner_tv
from gensim.models import Word2Vec
import numpy as np




def pridict(text):
    #word2vec_path = ner_tv.word2vec_path
    #word2vec = Word2Vec.load(word2vec_path)
    #sent_list = []
    #sent_length_list = []
    #for token in list(text):
        #if token not in word2vec.wv.vocab:
            #continue
        #sent_list.append(word2vec.wv.vocab[token].index)
    #sent_length_list.append(len(text))
    #if len(text) < ner_tv.max_length:
        #endChar_id = word2vec.wv.vocab["。"].index
        #sent_list += [endChar_id]*(ner_tv.max_length - len(text))




    model = lstm_crf.Model()
    with tf.Session() as sess:
        model.restore_model(sess)
        print(model.trains_params.eval())
        #test_unary_score, test_sequence_length = model.test_unary_score()
        #path = model.predict(sess,np.array([sent_list]),np.array(sent_length_list))
        #print(path)

        #new_sentence = model.out_put_sentences(path,list(text))
        #print(new_sentence)


if __name__ == '__main__':
    text = "这一点从王宏志的经历可以得到说明。"
    pridict(text)
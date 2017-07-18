# create by fanfan on 2017/7/14 0014

from src.ner_tv.lstm import bi_lstm_model_no_crf
import tensorflow as tf
from setting import ner_tv
from gensim.models import Word2Vec
import numpy as np
from jpype import *
startJVM(getDefaultJVMPath(), r"-Djava.class.path=E:/hanlp/hanlp-1.3.1.jar;E:/hanlp", "-Xms1g", "-Xmx1g")
HanLP = JClass("com.hankcs.hanlp.HanLP")


def pridict(text):
    word2vec_path = ner_tv.word2vec_path
    word2vec = Word2Vec.load(word2vec_path)
    sent_list = []
    sent_length_list = []

    segment_list = hanlp_list(text)
    for token in list(segment_list):
        if token not in word2vec.wv.vocab:
            continue
        sent_list.append(word2vec.wv.vocab[token].index)
    sent_length_list.append(len(segment_list))
    if len(segment_list) < ner_tv.max_length:
        endChar_id = word2vec.wv.vocab["。"].index
        sent_list += [endChar_id]*(ner_tv.max_length - len(segment_list))




    model = bi_lstm_model_no_crf.Model()
    with tf.Session() as sess:
        model.restore_model(sess)
        path = model.predict(sess,np.array([sent_list]),np.array(sent_length_list))
        print(path)

        new_sentence = model.out_put_sentences(path,list(text))
        print(new_sentence)

def CovertJlistToPlist(jList):
    ret = []
    if jList is None:
        return ret
    for i in range(jList.size()):
        ret.append(str(jList.get(i)))
    return ret

def hanlp_list(text):
    corpusText = HanLP.segment(text)
    corpus_list = CovertJlistToPlist(corpusText)
    listCorpus = list(corpus_list)
    new_list = []
    for index, item in enumerate(listCorpus):
        item_list = item.split("/")
        new_list.append(item_list[0] )
    return new_list

if __name__ == '__main__':
    text = "锦绣未央第五集"
    pridict(text)
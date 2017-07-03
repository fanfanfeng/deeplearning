__author__ = 'fanfan'
import codecs
from setting import ner
from gensim import corpora
import pickle
import random
import numpy as np
from gensim.models import Word2Vec

max_length = 50
def read_conll_file(path):
    """
    This function will Load sentences in Conll format.
    A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """


    sentences_x = []
    sentence_x = []

    sentences_y = []
    sentence_y = []

    word2vec_path = ner.word2vec_path
    word2vec = Word2Vec.load(word2vec_path)
    sentence_lengths = []

    endChar_id = word2vec.wv.vocab["。"].index

    for line in codecs.open(path,'r',encoding='utf-8'):
        line = line.rstrip().replace("\r","").replace("\n","")
        if not line:
            #用换行分割一句话
            sentence_len = len(sentence_x)
            if sentence_len >0:
                if sentence_len < max_length:
                    sentence_x += [endChar_id]*(max_length - sentence_len)
                    sentence_y += [0]*(max_length - sentence_len)
                sentences_x.append(sentence_x)
                sentences_y.append(sentence_y)
                sentence_lengths.append(sentence_len)

            sentence_x = []
            sentence_y = []
        else:
            words = line.split(" ")
            if len(words) > 1:
                if words[0] not in word2vec.wv.vocab:
                    continue
                if len(sentence_x) <50:
                    sentence_x.append(word2vec.wv.vocab[words[0]].index)
                    sentence_y.append(ner.tag_to_id[words[1]])
    if len(sentence_x) > 0:
        sentences_x.append(sentence_x)
        sentences_y.append(sentence_y)

    return  {"x":np.asarray(sentences_x),"y":np.asarray(sentences_y),"length":np.asarray(sentence_lengths)}


def load_word2Vec():

    word2vec_path = ner.word2vec_path
    word2vec = Word2Vec.load(word2vec_path)
    array_list = []
    for i in word2vec.wv.index2word:
        array_list.append(word2vec.wv[i])

    return np.array(array_list)

def make_vec_from_data(data):
    x = []
    y = []

    for item in data:
        x.append(item[0])
        temp =[]
        for y_label in item[1]:
            temp.append(ner.tag_to_id[y_label])
        y.append(temp)

    return {"x":x,"label":y}

class BatchManager():
    def __init__(self,data,batch_size):
        self.data_x = data["x"]
        self.data_y = data['y']
        self.data_each_sequence_length = data['length']
        self.len_data = data['length'].shape[0]
        self.batch_size = batch_size
        self.num_batch = self.len_data //batch_size

    def shuffle(self):
        random_index = np.random.permutation(np.arange(self.len_data))
        self.data_x = self.data_x[random_index]
        self.data_y = self.data_y[random_index]
        self.data_each_sequence_length = self.data_each_sequence_length[random_index]

    def iter_batch(self):
        for i in range(self.num_batch +1):
            if i == self.num_batch:
                data = {"x":self.data_x[i * self.batch_size:],"y":self.data_y[i*self.batch_size:],"lengths":self.data_each_sequence_length[i*self.batch_size:]}
            else:
                data = {"x":self.data_x[i * self.batch_size:(i+1) * self.batch_size],"y":self.data_y[i*self.batch_size:(i+1) * self.batch_size],"lengths":self.data_each_sequence_length[i*self.batch_size:(i+1) * self.batch_size]}
            yield data


def load_data():
    train_dict = read_conll_file(ner.training_path)
    dev_dict = read_conll_file(ner.dev_path)
    test_dict = read_conll_file(ner.test_path)

    return train_dict,dev_dict,test_dict


if __name__ == '__main__':
    train_data = read_conll_file(ner.training_path)
    #train_data = make_vec_from_data(train_data)
    #print(train_data.keys())
    for i in range(10):
        print(np.random.permutation(np.arange(10)))


# create by fanfan on 2017/7/11 0011
import tensorflow as  tf

from setting import ner_tv
import numpy as np
from gensim.models import  Word2Vec

def load_w2v():
    w2v_path = ner_tv.word2vec_path
    model = Word2Vec.load(w2v_path)
    array_list = []

    for i in model.wv.index2word:
        array_list.append(model.wv[i])
    return np.array(array_list)

def load_data():
    data_path = ner_tv.data_path

    word2vec_path = ner_tv.word2vec_path
    word2vec = Word2Vec.load(word2vec_path)

    endChar_id = word2vec.wv.vocab["。"].index

    x = []
    y = []
    lengths = []
    with open(data_path,'rb') as f:
        for index, line in enumerate(f):
            line = line.decode('utf-8').replace("\r","").replace("\n","")
            line_list = line.split(" ")

            item_x = []
            item_y = []
            for token in line_list:
                tag_x, tag_y = token.split("/")
                if tag_x not in word2vec.wv.vocab:
                    continue
                item_x.append(word2vec.wv.vocab[tag_x].index)
                tag_id = ner_tv.tag_to_id[tag_y]
                item_y.append(tag_id)

                if len(item_y) == ner_tv.flags.sentence_length:
                    break
            length = len(item_x)
            if len(item_x) < ner_tv.flags.sentence_length:
                item_x += [endChar_id] * (ner_tv.flags.sentence_length - len(item_x))
                item_y += [0] * (ner_tv.flags.sentence_length - len(item_y))

            if item_x and item_y and len(item_x) == len(item_y):
                x.append(item_x)
                y.append(item_y)
                lengths.append(length)
            else:
                print("error:{}".format(line))





    return {"x":np.array(x),"y":np.array(y),"length":np.array(lengths)}


class BatchManager():
    def __init__(self):
        self.sentenct_length = ner_tv.flags.sentence_length
        data_read = load_data()
        self.data_X = data_read['x']
        self.data_Y = data_read['y']
        self.lengths = data_read['length']
        self.totla_data_number = len(self.data_Y)
        shuffle_indices = np.random.permutation(np.arange(self.totla_data_number))
        x_shuffled = self.data_X[shuffle_indices]
        y_shuffled = self.data_Y[shuffle_indices]
        lengths_shuffled = self.lengths[shuffle_indices]

        # 分割训练集和测试机，用于验证

        valid_sample_index = int(self.totla_data_number * 0.1)
        self.x_valid, self.x_train = x_shuffled[:valid_sample_index], x_shuffled[valid_sample_index:]
        self.y_valid, self.y_train = y_shuffled[:valid_sample_index], y_shuffled[valid_sample_index:]
        self.lengths_valid,self.lengths_train  = lengths_shuffled[:valid_sample_index],lengths_shuffled[valid_sample_index:]

        self.train_data_num = len(self.y_train)
        self.batch_size = ner_tv.flags.batch_size
        self.num_batch = self.train_data_num // self.batch_size

    def shuffle(self):
        random_index = np.random.permutation(np.arange(self.train_data_num))
        self.x_train = self.x_train[random_index]
        self.y_train = self.y_train[random_index]
        self.lengths_train = self.lengths_train[random_index]

    def training_iter(self):
        self.shuffle()
        for i in range(self.num_batch +1):
            if i == self.num_batch:
                data = {"x":self.x_train[i * self.batch_size:],"y":self.y_train[i*self.batch_size:],"lengths":self.lengths_train[i*self.batch_size:]}
            else:
                data = {"x":self.x_train[i * self.batch_size:(i+1) * self.batch_size],"y":self.y_train[i*self.batch_size:(i+1) * self.batch_size],"lengths":self.lengths_train[i*self.batch_size:(i+1) * self.batch_size]}
            yield data


if __name__ == '__main__':
    #data = load_data()
    #b = data['x'][:100]
    #b = np.asarray(b,dtype=np.int32)
    #c = np.asarray(b)

    batch_manager = BatchManager()

    for batch in batch_manager.training_iter():
        test = batch_manager.x_train


        data_a_array = batch['x']

        print(np.asarray(data_a_array,dtype=np.int32))
        #print(a.shape)
        #print(b.shape)
        break
    #BatchManager()

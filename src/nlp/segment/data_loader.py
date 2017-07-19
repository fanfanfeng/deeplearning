# create by fanfan on 2017/7/11 0011
import tensorflow as  tf

from setting import nlp_segment as config_setting
import numpy as np
from gensim.models import  Word2Vec

#将gensim 的对象转化为一个 numpy array
def load_w2v():
    w2v_path = config_setting.word_vec_path
    model = Word2Vec.load(w2v_path)
    array_list = []

    for i in model.wv.index2word:
        array_list.append(model.wv[i])
    return np.asarray(array_list)


#加载数据，并转化为numpy array
def load_data():
    #获取训练数据所在目录
    data_path = config_setting.data_path
    #获取gensim word2vec所在目录
    word2vec_path = config_setting.word_vec_path
    word2vec = Word2Vec.load(word2vec_path)

    #句号的index,若果句子补助最大长度，用句号去补足
    endChar_id = word2vec.wv.vocab["。"].index

    inputs = []
    labels = []
    inputs_lengths = []
    with open(data_path,'rb') as f:
        for index, line in enumerate(f):
            line = line.decode('utf-8').replace("\r","").replace("\n","")
            line_list = line.split(" ")

            item_x = []
            item_y = []
            for token in line_list:
                try:
                    tag_x, tag_y = token.split("/")
                except:
                    continue
                if tag_x not in word2vec.wv.vocab:
                    print("{}不在word2vec里面".format(tag_x))
                    continue
                item_x.append(word2vec.wv.vocab[tag_x].index)
                tag_id = config_setting.tag_to_id[tag_y]
                item_y.append(tag_id)

                if len(item_y) == config_setting.flags.max_sentence_len:
                    break
            length = len(item_x)
            if len(item_x) < config_setting.flags.max_sentence_len:
                item_x += [endChar_id] * (config_setting.flags.max_sentence_len - len(item_x))
                item_y += [0] * (config_setting.flags.max_sentence_len - len(item_y))

            if item_x and item_y and len(item_x) == len(item_y):
                inputs.append(item_x)
                labels.append(item_y)
                inputs_lengths.append(length)
            else:
                print("error:{}".format(line))

    return {"inputs":np.asarray(inputs),"labels":np.asarray(labels),"inputs_length":np.asarray(inputs_lengths)}


#构造一个迭代对象
class BatchManager():
    def __init__(self):
        self.sentenct_length = config_setting.flags.max_sentence_len
        data_read = load_data()
        self.inputs = data_read['inputs']
        self.labels = data_read['labels']
        self.inputs_length = data_read['inputs_length']
        self.totla_data_number = len(self.inputs)

        shuffle_indices = np.random.permutation(np.arange(self.totla_data_number))
        inputs_shuffled = self.inputs[shuffle_indices]
        labels_shuffled = self.labels[shuffle_indices]
        inputs_length_shuffled = self.inputs_length[shuffle_indices]

        # 分割训练集和测试机，用于验证
        valid_sample_index = int(self.totla_data_number * 0.1)
        self.inputs_valid, self.inputs_train = inputs_shuffled[:valid_sample_index], inputs_shuffled[valid_sample_index:]
        self.labels_valid, self.labels_train = labels_shuffled[:valid_sample_index], labels_shuffled[valid_sample_index:]
        self.inputs_length_valid,self.inputs_length_train  = inputs_length_shuffled[:valid_sample_index],inputs_length_shuffled[valid_sample_index:]

        self.train_data_num = len(self.labels_train)
        self.batch_size = config_setting.flags.batch_size
        self.num_batch = self.train_data_num // self.batch_size

    def shuffle(self):
        random_index = np.random.permutation(np.arange(self.train_data_num))
        self.inputs_train = self.inputs_train[random_index]
        self.labels_train = self.labels_train[random_index]
        self.inputs_length_train = self.inputs_length_train[random_index]

    def training_iter(self):
        self.shuffle()
        for i in range(self.num_batch +1):
            if i == self.num_batch:
                data = {"inputs":self.inputs_train[i * self.batch_size:],"labels":self.labels_train[i*self.batch_size:],"inputs_lengths":self.inputs_length_train[i*self.batch_size:]}
            else:
                data = {"inputs":self.inputs_train[i * self.batch_size:(i+1) * self.batch_size],"labels":self.labels_train[i*self.batch_size:(i+1) * self.batch_size],"inputs_lengths":self.inputs_length_train[i*self.batch_size:(i+1) * self.batch_size]}
            yield data


if __name__ == '__main__':
    batch_manager = BatchManager()
    for batch in batch_manager.training_iter():


        data_a_array = batch_manager.labels

        print(data_a_array[0])
        #print(a.shape)
        #print(b.shape)
        break
    #BatchManager()

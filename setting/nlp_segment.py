__author__ = 'fanfan'
from setting import defaultPath
import os
import tensorflow as tf


#训练数据设置
data_path = os.path.join(defaultPath.PROJECT_DIRECTORY,'data/nlp/segment')

train_path = os.path.join(data_path,"train.txt")
test_path = os.path.join(data_path,"test.txt")
dev_path = os.path.join(data_path,'dev.txt')
#people 2014 data_path
people_2014 = os.path.join(data_path,'people2014/2014')



#词向量目录
word_vec_path = os.path.join(defaultPath.PROJECT_DIRECTORY,'model/zh_char_word2vec/vectors.model')
word2id_path = os.path.join(defaultPath.PROJECT_DIRECTORY,"model/zh_char_word2vec/word2id.pkl")

#模型保存位置
model_save_path = os.path.join(defaultPath.PROJECT_DIRECTORY,'model/nlp/segment')

#tf参数设置
tf.app.flags.DEFINE_integer('max_sentence_len',80,"一句话的的最大长度")
tf.app.flags.DEFINE_integer('embedding_size',100,'词向量的维度')
tf.app.flags.DEFINE_integer('num_tags',4,'BMES')
tf.app.flags.DEFINE_integer('num_hidden',200,'lstm 隐层个数')
tf.app.flags.DEFINE_integer('batch_size',100,'每次批量学习的大小')
tf.app.flags.DEFINE_integer('train_steps',50000,'训练总次数')
tf.app.flags.DEFINE_float('learning_rate',0.001,'学习率')
tf.flags.DEFINE_integer('max_grad_norm',5,'最大截断值')
tf.flags.DEFINE_float('dropout',0.5,'dropout的概率值')

tf.flags.DEFINE_integer('num_epochs',200,'重复训练的次数')
tf.flags.DEFINE_integer('valid_every',100,'每训练100次，在测试集上面验证模型')
tf.flags.DEFINE_integer("checkpoint_every", 200, "没训练200，保存模型")

#tag-id对应表
tag_to_id = {"S":0,"B":1,"M":2,"E":3}


flags = tf.flags.FLAGS

if __name__ == '__main__':
    flags._parse_flags()
    for key,values in flags.__flags.items():
        print("{}:{}".format(key,values))
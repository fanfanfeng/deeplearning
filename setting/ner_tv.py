# create by fanfan on 2017/7/10 0010
from setting import defaultPath
import os
import tensorflow as tf


source_type = 1 # 0, github ner 的wordvec model跟训练集
                # 1，ner_tv的 worvec model跟训练集
                # 2，ner_tv的字向量跟训练


if source_type == 0:
    tag_to_id = {"O": 0, "B-LOC": 1, "I-LOC": 2, "B-PER": 3, "I-PER": 4, "B-ORG": 5, "I-ORG": 6}
    word2vec_path = os.path.join(defaultPath.PROJECT_DIRECTORY, 'model/ner_tv/vectors.model')
    data_path = os.path.join(defaultPath.PROJECT_DIRECTORY, "data/ner_tv/out.txt")
    tf.flags.DEFINE_integer('tags_num', 7, '分类的数目')
    training_model_bi_lstm = os.path.join(defaultPath.PROJECT_DIRECTORY, "model/ner_tv/bilstm_train_model_github/")
elif source_type == 1:
    word2vec_path = os.path.join(defaultPath.PROJECT_DIRECTORY,'model/ner_tv/tv_data_word2vect.bin')
    tag_to_id = {"O": 0, "B-NAME": 1, "I-NAME": 2,"B-ARTIST": 3, "I-ARTIST": 4, "B-CATEGORY": 5, "I-CATEGORY": 6,"B-EPISODE":7,"I-EPISODE":8}
    data_path = os.path.join(defaultPath.PROJECT_DIRECTORY,"data/ner_tv/tagged_data.txt")
    tf.flags.DEFINE_integer('tags_num',9,'分类的数目')
    training_model_bi_lstm = os.path.join(defaultPath.PROJECT_DIRECTORY, "model/ner_tv/bilstm_train_model/")
elif source_type == 2:
    word2vec_path = os.path.join(defaultPath.PROJECT_DIRECTORY, 'model/ner_tv/single_vector.bin')
    tag_to_id = {"O": 0, "B-NAME": 1, "I-NAME": 2, "B-ARTIST": 3, "I-ARTIST": 4, "B-CATEGORY": 5, "I-CATEGORY": 6,
                 "B-EPISODE": 7, "I-EPISODE": 8}
    data_path = os.path.join(defaultPath.PROJECT_DIRECTORY, "data/ner_tv/single_tv_data.txt")
    tf.flags.DEFINE_integer('tags_num', 9, '分类的数目')
    training_model_bi_lstm = os.path.join(defaultPath.PROJECT_DIRECTORY, "model/ner_tv/bilstm_train_model/")


id_to_tag = { item[1]:item[0] for item in tag_to_id.items()}




max_length = 80


tf.flags.DEFINE_integer('embedding_dim',50,"词向量的维度")
tf.flags.DEFINE_integer('hidden_neural_size',256,'lstm隐层神经元数目')
tf.flags.DEFINE_integer('hidden_layer_num',1,'lstm的层数')
tf.flags.DEFINE_float('dropout',0.5,'dropout的概率值')
tf.flags.DEFINE_integer('batch_size',300,"每次批量学习的数目")
tf.flags.DEFINE_integer('sentence_length',80,'句子长度')
tf.flags.DEFINE_float("initial_learning_rate",0.01,'初始学习率')
tf.flags.DEFINE_float('min_learning_rate',0.0001,'最小学习率')
tf.flags.DEFINE_float('decay_rate',0.3,'学习衰减比例')
tf.flags.DEFINE_integer('decay_step',1000,'学习率衰减步长')
tf.flags.DEFINE_integer('max_grad_norm',5,'最大截断值')
tf.flags.DEFINE_integer('num_epochs',200,'重复训练的次数')
tf.flags.DEFINE_integer('show_every',10,'没训练10次，验证模型')
tf.flags.DEFINE_integer('valid_every',100,'每训练100次，在测试集上面验证模型')
tf.flags.DEFINE_integer("checkpoint_every", 200, "没训练200，保存模型")


flags = tf.flags.FLAGS
# create by fanfan on 2017/7/7 0007
import tensorflow as tf
import numpy as np
from setting import tv_classfication
from src.tv_classfication.bi_lstm import bi_lstm_model
import jieba
from gensim.models import Word2Vec
from common import  data_convert
def predict(text):
    words = jieba.cut(text)
    words = " ".join(words)
    index2label = {i: l.strip() for i, l in enumerate(tv_classfication.label_list)}

    word2vec_model = Word2Vec.load(tv_classfication.word2vec_path)
    text_converter = data_convert.SimpleTextConverter(word2vec_model, 80, None)
    x_test = []
    for doc, _ in text_converter.transform_to_ids([words]):
        x_test.append(doc)

    x_test = np.array(x_test)

    graph = tf.Graph()
    with graph.as_default(),tf.Session() as sess:
        model = bi_lstm_model.Bi_lstm()
        model.restore_model(sess)

        print(tv_classfication.index2label.get(model.predict(sess,x_test)[0]))








if __name__ == '__main__':
    text ="我想看成都的天气"
    predict(text)



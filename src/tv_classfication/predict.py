import  tensorflow as tf
import numpy as np
import jieba
from gensim.models import Word2Vec
from common import  data_convert

from setting import defaultPath,tv_classfication


def predict_label(text):

    words = jieba.cut(text)
    words = " ".join(words)


    word2vec_path = tv_classfication.word2vec_path
    model_save_path = "saveModel1/saveModel1/"
    word2vec_model = Word2Vec.load(word2vec_path)
    text_converter = data_convert.SimpleTextConverter(word2vec_model, 500, None)
    x_test = []
    for doc,_ in text_converter.transform_to_ids([words]):
        x_test.append(doc)

    x_test = np.array(x_test)

    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:


            checkpoint_file = tf.train.latest_checkpoint(model_save_path)
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            input_x = graph.get_operation_by_name("model/input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("model/dropout_keep_prob").outputs[0]

            logits = graph.get_operation_by_name("model/softmax_layer/Add:0").outputs[0]



            prediction = graph.get_operation_by_name("model/output/prediction").outputs[0]
            predict_class,logits_out = sess.run([prediction,logits], {input_x: x_test, dropout_keep_prob: 1.0})[0]
            return tv_classfication.index2label.get(predict_class)

if __name__ == '__main__':
    text = ""
    print(predict_label(text))

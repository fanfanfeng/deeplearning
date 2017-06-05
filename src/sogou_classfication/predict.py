import tensorflow as tf
import numpy as np
import jieba,os

from setting import defaultPath,sogou_classfication
from common import data_convert,stop_word
from gensim.models import Word2Vec

mode_save_path = os.path.join(defaultPath.PROJECT_DIRECTORY,sogou_classfication.lstm_model_save_path,"saveModel")

# checkpoint_dir,训练时保存的模型
tf.flags.DEFINE_string('checkpoint_dir',mode_save_path,"model path")
# max_sentence_length,文本最大长度
tf.flags.DEFINE_integer('max_sentence_length',500,'max sentence length')


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("Parameters:")
for attr,value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(),value))

print("")

def predict_doc(text):
    """
    给定一个文本，预测文本的分类
    """
    stop_word_set = stop_word.get_stop_word()
    sengment_list = jieba.cut(text)
    word_list = []
    for word in sengment_list:
        word = word.strip()
        if ''!=word and word not in stop_word_set:
            word_list.append(word)

    word_segment = " ".join(word_list)


    #查找最新保存的检查文件的文件名
    checkpoint_dir = FLAGS.checkpoint_dir
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    index2label_dict = {i:l.strip() for i,l in enumerate(sogou_classfication.label_list)}

    model_path = os.path.join(defaultPath.PROJECT_DIRECTORY, sogou_classfication.word2Vect_path)
    model_save_path = os.path.join(model_path, sogou_classfication.model_name)
    word2vec_model = Word2Vec.load(model_save_path)
    text_converter = data_convert.SimpleTextConverter(word2vec_model, FLAGS.max_sentence_length, None)

    x_test = []
    for doc,_ in text_converter.transform_to_ids([word_segment]):
        x_test.append(doc)
    x_test = np.array(x_test)


    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess,checkpoint_file)
            input_x = graph.get_operation_by_name("model/input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("model/dropout_keep_prob").outputs[0]
            #带评估的Tensors
            prediction = graph.get_operation_by_name("model/output/prediction").outputs[0]
            predict_class = sess.run(prediction,{input_x:x_test,dropout_keep_prob:1.0})[0]
            return index2label_dict.get(predict_class)

if __name__ == '__main__':
    text="""
    三九医药 （ 000999 ） 和 三九生化 （ 000403 ） 今日 同时 发布公告 ， 三九医药 转让 三九生化 38.11 ％ 股权 事宜 获得 国资委 批准 
    """
    print(predict_doc(text))


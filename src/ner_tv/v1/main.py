# create by fanfan on 2017/7/26 0026
from setting import ner_tv
from src.ner_tv.v1 import data_loader
import tensorflow as tf
from src.ner_tv.v1 import blstm_crf

def train():
    train_data_manager = data_loader.BatchManager(ner_tv.train_path, ner_tv.flags.batch_size)
    test_data_manager = data_loader.BatchManager(ner_tv.test_path, ner_tv.flags.batch_size)

    with tf.Session() as sess:
        model = blstm_crf.Model()
        model.model_restore(sess)

        for epoch in range(model.train_epoch):
            print("start epoch {}".format(str(epoch)))
            for batch in train_data_manager.training_iter():
                train_inputs = batch['sentence_words_id']
                train_labels = batch['sentence_tags_id']
                step,loss = model.run_step(sess,train_inputs,train_labels,True)
                if step % 100 == 0:
                    print("iteration:{} step:{},NER loss:{:>9.6f}".format(epoch,  step, loss))




            total_accuracy = 0
            for batch in test_data_manager.training_iter():
                test_inputs = batch['sentence_words_id']
                test_labels = batch['sentence_tags_id']
                accuracy = model.test_accuraty(sess,test_inputs,test_labels)
                total_accuracy += accuracy

            mean_accuracy = total_accuracy/test_data_manager.num_batch
            print("iteration:{},NER accuracy:{:>9.6f}".format(epoch, mean_accuracy))
            model.saver.save(sess, model.model_save_path, global_step=epoch)


import pickle
import os
import numpy as np
def predict(text):
    if os.path.exists(ner_tv.dict_word2vec_path):
        word2id_dict = pickle.load(open(ner_tv.dict_word2vec_path,'rb'))
        words_list = list(text)
        words_list_id = [word2id_dict[i] for i in words_list]
        text_len = len(words_list_id)
        if text_len <80:
            words_list_id += [0] * (80 - text_len)

        inputs = np.array(words_list_id).reshape([1, 80])

        with tf.Session() as sess:
            model = blstm_crf.Model()
            model.model_restore(sess)

            fenchiResult = {
                "name":"",
                "artist":"",
                "category":"",
                "episode":""
            }

            path = model.predict(sess,inputs)
            for word,seg_id in zip(words_list,path[0]):
                if seg_id == 1:
                    fenchiResult["name"] += word
                elif seg_id == 2:
                    fenchiResult["name"] += word + " "
                elif seg_id == 3:
                    fenchiResult["artist"] += word
                elif seg_id == 4:
                    fenchiResult["artist"] += word + " "
                elif seg_id == 5:
                    fenchiResult["category"] += word
                elif seg_id == 6:
                    fenchiResult["category"] += word + " "
                elif seg_id == 7:
                    fenchiResult["episode"] += word
                elif seg_id == 8:
                    fenchiResult["episode"] += word + " "

            return fenchiResult






if __name__ == '__main__':
    #train()
    text = u'我想看李连杰的电视剧楚乔传第3集'
    result = predict(text)
    print("分词前：" + text)
    print("分词后：" )
    for key in result:
        print("{} : {}".format(key,result[key]))

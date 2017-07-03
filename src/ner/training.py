__author__ = 'fanfan'
import os
import tensorflow as tf
from setting import ner
from src.ner import prepare_data
from src.ner import bi_lstm_model
from tensorflow.contrib import crf
import numpy as np

FLAGS = tf.flags.FLAGS

#data path setting
tf.flags.DEFINE_string("train_file",ner.training_path,'path for training data')
tf.flags.DEFINE_string('valid_file',ner.data_path,'path for valid data')
tf.flags.DEFINE_string('test_file',ner.test_path,'path for test data')
tf.flags.DEFINE_string('word2vec_path',ner.word2vec_path,'paht for word2vec model ,trained by gensim ')
tf.app.flags.DEFINE_string("model_path", 'modelSave/', "path to save model")

#model setting
tf.flags.DEFINE_integer('min_freq',2,"minimum count of word")
tf.flags.DEFINE_integer('sentence_max_len',50,'maxium words in a sentence')
tf.flags.DEFINE_integer('input_dim',50,"word vec size")
tf.flags.DEFINE_integer('neaurl_hidden_dim',256,"dimension of the lstm hidden nuits")
tf.flags.DEFINE_string('feature_dim',0,'dimension of extra features,0 for not used')

#config for training
tf.flags.DEFINE_float('dropout',0.5,'dropout rate')
tf.flags.DEFINE_float('clip',5,'gradient to clip')
tf.flags.DEFINE_float('learning_rate',0.0001,'initial learning rate')
tf.flags.DEFINE_integer('max_epoch',150,'maxinum training epochs')
tf.flags.DEFINE_integer('batch_size',20,'num of sentences per batch')
tf.flags.DEFINE_integer("steps_per_checkpoint",100,'steps per checkpoint')
tf.flags.DEFINE_integer("valid_batch_size",100,'num of sentences per batch')


if __name__ == '__main__':
    word2vec = prepare_data.load_word2Vec()
    train_data,dev_data,test_data = prepare_data.load_data()
    traning_manager = prepare_data.BatchManager(train_data,FLAGS.batch_size)
    dev_manager = prepare_data.BatchManager(dev_data,FLAGS.batch_size)
    test_manager = prepare_data.BatchManager(test_data,FLAGS.batch_size)


    print("loading data over .......")
    with tf.Session() as sess:
        id_to_tag = {v: k for k, v in ner.tag_to_id.items()}
        model = bi_lstm_model.model("ner_tag",id_to_tag,FLAGS,word2vec)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            last_checkpoint = tf.train.lastest_checkpoint(FLAGS.model_path)
            saver.restore(sess,last_checkpoint)
        else:
            sess.run(tf.global_variables_initializer())

        loss = 0
        best_test_f1 = 0
        steps_per_epoch = int(traning_manager.len_data/FLAGS.batch_size)  + 1
        for i in range(FLAGS.max_epoch):
            print("Start {} epoch training".format(i))
            iteration = (model.global_step.eval()) // steps_per_epoch + 1
            traning_manager.shuffle()
            for batch in traning_manager.iter_batch():
                global_step = model.global_step.eval()
                step = global_step % steps_per_epoch
                feed_dict = {
                    model.inputs:batch["x"],
                    model.labels:batch["y"],
                    model.dropout:FLAGS.dropout,
                    #model.features:0,
                    model.lengths:batch['lengths'],
                }
                loss,_ =  sess.run([model.loss, model.train_op],feed_dict = feed_dict)

                if global_step % FLAGS.steps_per_checkpoint == 0:
                    print("iteraiton:{} step :{}/{},ner loss:{:>9.6f}".format(iteration,step,steps_per_epoch,loss))


            #predict
            correct_num = 0
            total_num = 0

            trans = model.transiton_params.eval()
            for batch in dev_manager.iter_batch():
                feed_dict = {
                    model.inputs:batch["x"],
                    model.labels:batch["y"],
                    model.dropout:1,
                    #model.features:0,
                    model.lengths:batch["lengths"],
                }

                #获取状态
                scores = sess.run(model.scores,feed_dict)

                #viterbi 解码返回最可能的路径

                for score,length,y_ in zip(scores,batch["lengths"],batch['y']):
                    score = score[:length]
                    path,_ = crf.viterbi_decode(score,trans)

                    #evaluate word-level accuracy
                    correct_num += np.sum(np.equal(path,y_[:length]))
                    total_num += length
            correct_rate = 100 * correct_num/total_num
            print(" dev correcnt rate: {:.2f}%%".format(correct_rate))

        #predict
        correct_num = 0
        total_num = 0

        trans = model.transiton_params.eval()
        for batch in dev_manager.iter_batch():
            feed_dict = {
                model.inputs:batch['x'],
                model.labels:batch['y'],
                model.dropout:1,
                model.features:0,
                model.lengths:batch["lengths"],
            }

            #获取状态
            scores = sess.run(model.scores,feed_dict)

            #viterbi 解码返回最可能的路径

            for score,length,y_ in zip(scores,batch["lengths"],batch['y']):
                score = score[:length]
                path,_ = crf.viterbi_decode(score,trans)

                #evaluate word-level accuracy
                correct_num += np.sum(np.equal(path,y_))
                total_num += length
        correct_rate = 100 * correct_num/total_num
        print("end test correcnt rate: {:.2f }%%".format(correct_rate))




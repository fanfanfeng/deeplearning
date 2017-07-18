# create by fanfan on 2017/7/12 0012
import tensorflow as tf

from setting import ner_tv
from src.ner_tv import data_loader
from src.ner_tv.lstm import bi_lstm_model_no_crf


def train():
    data_dict = data_loader.BatchManager()
    graph = tf.Graph()
    with graph.as_default(),tf.Session() as sess:
        model = bi_lstm_model_no_crf.Model()
        try:
            model.restore_model(sess)
        except:
            init = tf.global_variables_initializer()
            sess.run(init)

        for num_poch in range(ner_tv.flags.num_epochs):
            print("第{}轮训练 ".format(num_poch + 1))
            for batch_data in data_dict.training_iter():
                if len(batch_data) ==0:
                    continue

                train_input = batch_data['x']
                train_label = batch_data['y']
                train_lengths = batch_data['lengths']
                step,_, = model.train_step(sess,train_input,train_label,train_lengths,is_training=True)
                if step % ner_tv.flags.show_every == 0:
                    _,loss = model.train_step(sess,train_input,train_label,train_lengths,is_training=True)
                    print("第{}轮训练 ，第{}步，损失值是{:g}".format(num_poch+1,step,loss))

                if step % ner_tv.flags.valid_every == 0:
                    accuracy = model.train_step(sess,train_input,train_label,train_lengths,is_training=False)
                    print("第{}轮训练 ，第{}步，准确值是{:g}".format(num_poch + 1, step, accuracy))

            step = model.global_step.eval()
            accuracy = model.test_accurate(sess,data_dict.x_valid,data_dict.y_valid,data_dict.lengths_valid)
            print("第{}轮训练 ，第{}步，准确率是{:g}".format(num_poch + 1, step, accuracy))

            model.saver.save(sess,model.model_save_path,step)


if __name__ == '__main__':
    train()



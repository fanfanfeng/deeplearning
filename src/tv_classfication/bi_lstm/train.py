# create by fanfan on 2017/7/4 0004
import os
import sys
import numpy as np
import tensorflow as tf
from setting import tv_classfication
from setting import defaultPath
from src.tv_classfication.bi_lstm import  bi_lstm_model
from src.tv_classfication.bi_lstm import tv_data_loader


def train():

    tv_data = tv_data_loader.tv_data_loader()
    graph = tf.Graph()
    with graph.as_default(),tf.Session() as sess:
        model = bi_lstm_model.Bi_lstm()

        check_point = tf.train.get_checkpoint_state(model.train_model_save_path)
        if check_point and tf.train.checkpoint_exists(check_point.model_checkpoint_path):
            print("reading model from %s" % check_point.model_checkpoint_path)
        else:
            print("create model ")
            sess.run(tf.global_variables_initializer())


        for num_epoch in range(tv_classfication.flags.num_epochs):
            print("epoch  {}".format(num_epoch +1))
            for batch_data in tv_data.training_iter():
                if len(batch_data) == 0:
                    continue
                train_x,train_y = zip(*batch_data)
                step,_,_,_ = model.train_step(sess,is_train=True,inputX= train_x,inputY = train_y)
                if step % tv_classfication.flags.show_every == 0:
                    _,learning_rate,loss,accuracy = model.train_step(sess,is_train=True,inputX= train_x,inputY = train_y)
                    print("第{}轮训练, 训练步数 {}, 学习率 {:g}, 损失值 {:g}, 精确值 {:g}".format(num_epoch + 1, step, learning_rate, loss, accuracy))

                if step % tv_classfication.flags.valid_every == 0:
                    _,learning_rate,loss,accuracy = model.train_step(sess,False,tv_data.x_valid,tv_data.y_valid)
                    print("验证模型, 训练步数 {} ,学习率 {:g}, 损失值 {:g}, 精确值 {:g}".format(step, learning_rate, loss, accuracy))


                if step % tv_classfication.flags.checkpoint_every == 0:
                    path = model.saver.save(sess,tv_classfication.train_model_bi_lstm,step)
                    print("模型保存到{}".format(path))

if __name__ == '__main__':
    train()






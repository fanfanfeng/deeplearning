import numpy as np
import tensorflow as tf
from  setting import defaultPath,sogou_classfication
from src.sogou_classfication import lstm_model
import os

def load_data(max_sentence_length = None):
    """
        从本地文件读取搜狗分类数据集
    """
    read_dir_path = os.path.join(defaultPath.PROJECT_DIRECTORY,sogou_classfication.data_path_jieba)
    label_dir_list = os.listdir(read_dir_path)
    x_raw = []
    y = []
    label2index_dict = {l.strip(): i for i,l in enumerate(sogou_classfication.label_list)}

    for label_dir in label_dir_list:
        label_dir_path = os.path.join(read_dir_path,label_dir)
        label_index = label2index_dict[label_dir]
        label_item = np.zeros(len(sogou_classfication.label_list),np.float32)
        label_item[label_index] = 1
        label_file_list = os.listdir(label_dir_path)
        for label_file in label_file_list:
            with open(os.path.join(label_file_list,label_file),'rb') as reader:
                text = reader.read().decode('utf-8').replace('\n','').replace('\r','').strip()
                x_raw.append(text)
                y.append(label_item)
        if not max_sentence_length:
            max_sentence_length = max([len(item.split(" ") for item in x_raw)])
        x = []



__author__ = 'fanfan'
from setting import defaultPath
import os
tv_data_path = os.path.join(defaultPath.PROJECT_DIRECTORY,"data/tv_classfication")
thc_data_path = r"E:\github\deeplearning\THC\THUCNews"

word2vec_path = os.path.join(defaultPath.PROJECT_DIRECTORY,"model/word2vec/w2v.model")
label_list = ['app', 'chat', 'joke', 'music', 'player', 'video', 'weather']


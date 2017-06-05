import os,sys

from setting import defaultPath,sogou_classfication
from gensim.models import Word2Vec

def training_word2vec():
    sentences = []

    read_dir_path = os.path.join(defaultPath.PROJECT_DIRECTORY,sogou_classfication.data_path_jieba)
    label_dir_list = os.listdir(read_dir_path)
    for label_dir in label_dir_list:
        label_dir_path = os.path.join(read_dir_path,label_dir)
        label_file_list = os.listdir(label_dir_path)
        for label_file in label_file_list:
            with open(os.path.join(label_dir_path,label_file),'rb') as reader:
                word_list = reader.read().decode('utf-8').replace('\n','').replace('\r','').strip()
                sentences.append(word_list)

    model_path = os.path.join(defaultPath.PROJECT_DIRECTORY,sogou_classfication.word2Vect_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_save_path = os.path.join(model_path,sogou_classfication.model_name)

    model = Word2Vec(sentences,max_vocab_size=None,window=8,size=256,min_count=5,workers=4,iter=20)
    model.save(model_save_path)



if __name__ == '__main__':
    training_word2vec()
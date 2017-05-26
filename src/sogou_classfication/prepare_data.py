import os,sys

currentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if currentDir not in sys.path:
    sys.path.append(currentDir)

from setting import defaultPath,sogou_classfication
from common import stop_word

#将文本中的句子用 结巴分词进行分词 用空格分开
def segment_data():
    stopwords_set = stop_word.get_stop_word()







def train_sogou_classfication_word2vec():

    sentences = []
    read_dir_path = os.path.join(defaultPath.PROJECT_DIRECTORY,sogou_classfication.data_path)

    label_dir_list = os.listdir(read_dir_path)
    for label_dir in label_dir_list:
        label_dir_path = os.path.join(read_dir_path,label_dir)
        label_file_list = os.listdir((label_dir_path))
        for label_file in label_file_list:
            with open(os.path.join(label_dir_path,label_file),'r') as reader:
                word_list = reader.read().replace('\n', '').replace('\r', '').strip().split()
                sentences.append(word_list)
                break
            break

    print(sentences)


train_sogou_classfication_word2vec()
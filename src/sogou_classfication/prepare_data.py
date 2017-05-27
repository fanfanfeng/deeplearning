import os,sys

currentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if currentDir not in sys.path:
    sys.path.append(currentDir)

from setting import defaultPath,sogou_classfication
from common import stop_word
import jieba

#将文本中的句子用 结巴分词进行分词 用空格分开
def segment_data():
    stopwords_set = stop_word.get_stop_word()
    return stopwords_set

def fenchi_jieba():
    stopwords_set = segment_data()
    #读取文件目录
    read_dir_path = os.path.join(defaultPath.PROJECT_DIRECTORY, sogou_classfication.data_path_origin)
    write_dir_path = os.path.join(defaultPath.PROJECT_DIRECTORY,sogou_classfication.data_path_jieba)

    #不存在写目录的话，就创建该目录
    if not os.path.exists(write_dir_path):
        os.makedirs(write_dir_path)

    #读取目录下面的所有标签
    label_dir_list = os.listdir(read_dir_path)
    for label in label_dir_list:

        #标签具体路径
        label_read_dir = os.path.join(read_dir_path,label)
        label_write_dir = os.path.join(write_dir_path,label)

        #不存在标签目录就创建
        if not os.path.exists(label_write_dir):
            os.makedirs(label_write_dir)

        label_file_list = os.listdir(label_read_dir)
        process_num = 0
        for label_file in label_file_list:
            process_num += 1
            label_file_read_path = os.path.join(label_read_dir,label_file)
            label_file_write_path = os.path.join(label_write_dir,label_file)

            write_file = open(label_file_write_path,'wb')
            with open(label_file_read_path,'rb') as reader:
                text = reader.read().decode('utf-8').replace("\n","").replace('\r',"").strip()
                segment_list = jieba.cut(text)
                word_list = []
                for word in segment_list:
                    word =  word.strip()
                    if '' != word and word not in stopwords_set:
                        word_list.append(word)

                word_str = " ".join(word_list)
                write_file.write(word_str.encode('utf-8'))
            write_file.close()
        print("success file num:%s" % process_num)

#train_sogou_classfication_word2vec()
if __name__ == '__main__':
    fenchi_jieba()
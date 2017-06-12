__author__ = 'fanfan'
import  os
from setting import tv_classfication
from gensim.models import Word2Vec
from multiprocessing import process
class data_work():
    def __init__(self,path):
        self.path = path


    def __iter__(self):
        for path,dirs,files in os.walk(self.path):
            if len(files) == 0:
                continue

            for file in files:
                real_file_path = os.path.join(path,file)
                with open(real_file_path,'rb') as reader:
                    for line in reader:
                        line = line.decode('utf-8')
                        line = line.split(" ")
                        if len(line) > 128:
                            line = line[:128]
                        yield line

def make_word2vec():
    data_path = tv_classfication.tv_data_path
    sentence = data_work(data_path)
    model = Word2Vec(sentence,size=256,workers=4,window=10,iter=30)
    model.save(tv_classfication.word2vec_path)


if __name__ == '__main__':
    make_word2vec()
__author__ = 'fanfan'
from setting import tv_classfication
from gensim.models import Word2Vec

if __name__ == '__main__':
    model_path = tv_classfication.word2vec_path
    model = Word2Vec.load(model_path)
    print(model['你好'])
    print(model.most_similar(positive=['女人', '王子'], negative=['男人'], topn=1))
    print(model.most_similar(['男人']))
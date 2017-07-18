# create by fanfan on 2017/7/10 0010
from gensim.models import Word2Vec
from setting import ner_tv

model = Word2Vec.load(ner_tv.word2vec_path)
print(model.wv.vocab["。"])
print(model.wv.vocab["你"])
print()


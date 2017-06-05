from gensim import corpora,models,similarities
from setting import defaultPath
import os

corpus_store_path = os.path.join(defaultPath.PROJECT_DIRECTORY,"data/word2vec_result/deerwester.mm")
corpus = corpora.MmCorpus(corpus_store_path)
print(corpus)

tfidf = models.TfidfModel(corpus)
print(tfidf)

doc_bow = [(0,1),(1,1)]
print(tfidf[doc_bow])

corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)


path_store = os.path.join(defaultPath.PROJECT_DIRECTORY,"data/word2vec_result/deerwester.dict")
dictionary = corpora.Dictionary.load(path_store)

lsi = models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=2)
corpus_lsi = lsi[corpus_tfidf]

print(lsi.print_topics(2))

for i,doc in  enumerate(corpus_lsi):
    print(doc)

path_store = os.path.join(defaultPath.PROJECT_DIRECTORY,"data/word2vec_result/model.lsi")

lsi.save(path_store)
lsi = models.LsiModel.load(path_store)
from gensim import corpora,models,similarities
from setting import defaultPath
import os


dict_path = os.path.join(defaultPath.PROJECT_DIRECTORY,"data/word2vec_result/deerwester.dict")
dictionary = corpora.Dictionary.load(dict_path)

mm_path = os.path.join(defaultPath.PROJECT_DIRECTORY,"data/word2vec_result/deerwester.mm")
corpus = corpora.MmCorpus(mm_path)
print(corpus)

lsi = models.LsiModel(corpus,id2word=dictionary,num_topics=2)

doc = 'Human computer interaction'
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]
print(vec_lsi)

index = similarities.MatrixSimilarity(lsi[corpus])

index_save = os.path.join(defaultPath.PROJECT_DIRECTORY,"data/word2vec_result/deerwester.index")
index.save(index_save)
index = similarities.MatrixSimilarity.load(index_save)

print(index)

sims = index[vec_lsi]
print(list(enumerate(sims)))


sims = sorted(enumerate(sims),key= lambda item:-item[1])
print(list(enumerate(sims)))
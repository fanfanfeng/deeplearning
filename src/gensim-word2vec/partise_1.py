import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
from gensim import corpora,models,similarities
corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
          [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
          [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
          [(0, 1.0), (4, 2.0), (7, 1.0)],
          [(3, 1.0), (5, 1.0), (6, 1.0)],
          [(9, 1.0)],
          [(9, 1.0), (10, 1.0)],
          [(9, 1.0), (10, 1.0), (11, 1.0)],
          [(8, 1.0), (10, 1.0), (11, 1.0)]]

tfidf = models.TfidfModel(corpus)
print(tfidf)

#一个变换，可以将文档从一个向量表示变换成另外的一个:
vec = [(0,1),(4,1)]
print(tfidf[vec])

#整个语料可以通过tfidf进行变换和索引，相似查询
index = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=12)
print(index)
#我们将待查询的矢量vec，对在语料库中的每个文档进行相似查询相似.
sims = index[tfidf[vec]]
print(list(enumerate(sims)))
#文档号0（第一个文档）具有的相似分为：0.466=46.6%，第二个文档具有相似分:19.1% 等。

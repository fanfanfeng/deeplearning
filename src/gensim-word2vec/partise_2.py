from gensim import  corpora,models,similarities
from setting import defaultPath
import gensim
import os
#这是一个小语料，由9个文档组成，每个文档都由一句话组成.
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[ word for word in document.lower().split() if word not in stoplist] for document in documents]

print(texts)

# remove words that appear only once
all_tokens = sum(texts,[])
print(all_tokens)
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once] for text in texts]

print(texts)

dictionary = corpora.Dictionary(texts)
path_store = os.path.join(defaultPath.PROJECT_DIRECTORY,"data/word2vec_result/deerwester.dict")
dictionary.save(path_store)
print(dictionary)

#词与id之间的映射关系
print(dictionary.token2id)

#将切割过的文档转换成向量
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
# the word "interaction" does not appear in the dictionary and is ignored
print(new_vec)

corpus = [dictionary.doc2bow(text) for text in texts]
corpus_store_path = os.path.join(defaultPath.PROJECT_DIRECTORY,"data/word2vec_result/deerwester.mm")
corpora.MmCorpus.serialize(corpus_store_path,corpus)
print(corpus)

#用迭代器来读文件，文件很大的时候，省内存
class MyCorpus(object):
    def __iter__(self):
        for line in open('mycorpus.txt','rb'):
            yield dictionary.doc2bow(line.decode('utf-8').lower().split())


dictionary = corpora.Dictionary(line.decode('utf-8').lower().split() for line in open('mycorpus.txt','rb'))

stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]

once_ids = [tokenid for tokenid,docfreq in dictionary.dfs.items() if docfreq==1]
# remove stop words and words that appear only once
dictionary.filter_tokens(stop_ids + once_ids)
#remove gaps in id sequence after words that were removed
dictionary.compactify()
print(dictionary)



#预料保存

corpus = [[(1,0.5)],[]]
#mm(Market Matrix)格式
corpora.MmCorpus.serialize('corpus.mm',corpus)
#joachim的SVMlight格式
corpora.SvmLightCorpus.serialize('corpus.svmlight',corpus)
#Blei的LDA-C格式
corpora.BleiCorpus.serialize('corpus.lda-c',corpus)
#GibbsLDA++
corpora.LowCorpus.serialize('corpus.low',corpus)

#从一个MM文件中加载一个语料
corpus = corpora.MmCorpus('corpus.mm')
print(corpus)
#查看语料中的内容
print(list(corpus))

for doc in corpus:
    print(doc)



#4.兼容NumPy和SciPy
numpy_matrix = gensim.matutils.corpus2dense(corpus,12)
print(numpy_matrix)
corpus = gensim.matutils.Dense2Corpus(numpy_matrix)
print(corpus)

#scipy.sparse矩阵的from/to函数
scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)
print(scipy_csc_matrix)
corpus = gensim.matutils.Sparse2Corpus(scipy_csc_matrix)
print(corpus)










from setting import defaultPath,sogou_classfication
from gensim.models import Word2Vec
import os
import numpy as np

model_path = os.path.join(defaultPath.PROJECT_DIRECTORY, sogou_classfication.word2Vect_path)
model_save_path = os.path.join(model_path, sogou_classfication.model_name)
word2vec_model = Word2Vec.load(model_save_path)

print(word2vec_model.wv.vocab)
print(word2vec_model.wv.index2word)

print(word2vec_model.wv["的"])
print(word2vec_model['的'])

array_list = []
for i in word2vec_model.wv.index2word:
    array_list.append(word2vec_model.wv[i])
print(array_list[3])
print(np.array(array_list).shape)



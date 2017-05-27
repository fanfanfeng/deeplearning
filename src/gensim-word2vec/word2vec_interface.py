from gensim.models import Word2Vec
from gensim import matutils


def train_word2vec_by_gensim(sentences,model_name):
    """
        训练word2vec
        :param sentences: 分词列表,例如[['first', 'sentence'], ['second', 'sentence']]
        :param model_name: 保存Word2Vec的模型名
        :return:
    """

    model = Word2Vec(sentences = sentences,max_vocab_size=None,window=8,size=256,min_count=5,workers=4,iter=20)
    model.save(model_name)


def get_word_2_vec_by_gensim(model_name):
    #获取model_name对应的word2vec
    word2vec_by_gensim = Word2Vec.load(model_name)
    word2vec_by_gensim.init_sims(replace=True)
    return word2vec_by_gensim



def get_word_most_similar_words_in_word2vec_gensim(modelname,word,top=10):
    #计算某个词的相关词列表
    model = get_word_2_vec_by_gensim(model_name=modelname)
    return model.most_similar(word,topn=top)

def get_similar_sentences_in_word2vec_gensim(model_name,sentence_1,sentence_2):
    #计算两个句子之间的相似度/相关程度
    model = get_word_2_vec_by_gensim(model_name)
    return model.n_similarity(sentence_1,sentence_2)
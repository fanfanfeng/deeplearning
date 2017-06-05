from common import  constant
import numpy as np
def default_tokenizer():
    return lambda x: x.split()

class SimpleTextConverter(object):
    def __init__(self,word_vec,max_document_length,tokenizer_fn=None):
        #self.syn0norm = word_vec.syn0norm
        self.vocab = word_vec.wv.vocab
        self.tokenizer_fn = tokenizer_fn or default_tokenizer()
        self.max_document_length = max_document_length


    def transform_to_ids(self,raw_documents):
        for text in raw_documents:
            tokens = self.tokenizer_fn(text)
            word_ids = np.zeros(self.max_document_length,np.int64)

            idx = 0
            for token in tokens:
                if token not in self.vocab:
                    continue

                if idx >= self.max_document_length:
                    break

                word_ids[idx] = self.vocab[token].index
                idx +=1

            yield word_ids,idx


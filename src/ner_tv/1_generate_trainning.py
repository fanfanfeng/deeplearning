import sys
import os
from setting import  ner_tv
from gensim.models import Word2Vec

totalLine = 0
longLine = 0

MAX_LEN = 80
totalChars = 0


if __name__ == "__main__":
    global totalChars
    global longLine
    global totalLine

    outPath = ner_tv.data_path

    test_write = open(os.path.join(os.path.dirname(outPath),'test.txt'),'wb')
    train_write = open(os.path.join(os.path.dirname(outPath),'train.txt'),'wb')

    word2vec_path = ner_tv.word2vec_path
    word2vec = Word2Vec.load(word2vec_path)

    with open(outPath,'rb') as fread:
        no = 0
        for line in fread:
            line = line.decode('utf-8')
            save_list = [0]*160
            tokens_list = line.split(" ")
            for index, token in enumerate(tokens_list):
                words_list = token.split('/')
                save_list[index] = word2vec.wv.vocab[words_list[0]].index
                save_list[index +80] = ner_tv.tag_to_id[words_list[1]]

            if no %8 ==0:
                test_write.write((" ".join(save_list) + "\n").encode("utf-8"))
            else:
                train_write.write((" ".join(save_list) + "\n").encode("utf-8"))
            no += 1

    test_write.close()
    train_write.close()



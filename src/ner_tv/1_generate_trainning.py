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

            if no %8 ==0:
                test_write.write((line).encode("utf-8"))
            else:
                train_write.write((line).encode("utf-8"))
            no += 1

    test_write.close()
    train_write.close()



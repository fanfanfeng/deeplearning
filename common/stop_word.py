from setting import defaultPath,sogou_classfication
import os,sys

def get_stop_word():
    stop_words = set()
    with open(os.path.join(defaultPath.PROJECT_DIRECTORY, defaultPath.Stop_Words),'rb') as reader:
        for each_line in reader:
            word = each_line.decode('utf-8').replace('\n', '')
            stop_words.add(word)

    return stop_words
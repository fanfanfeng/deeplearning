from setting import defaultPath,sogou_classfication
import os,sys

def get_stop_word():
    stop_words = set()
    with open(os.path.join(defaultPath.PROJECT_DIRECTORY, defaultPath.Stop_Words)) as reader:
        for each_line in reader.readlines():
            word = each_line.replace('\n', '')
            stop_words.add(word)

    return stop_words
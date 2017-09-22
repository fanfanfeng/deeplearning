__author__ = 'fanfan'
from collections import Counter
import random
import os
import sys
import zipfile

import numpy as np
import tensorflow as tf
import urllib.request
import codecs


DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
EXPECTED_BYTES = 31344016
DATA_FOLDER = 'data/'
FILE_NAME = 'text8.zip'

def download(file_name,expected_bytes):
    file_path = DATA_FOLDER + file_name
    if os.path.exists(file_path):
        print("Dataset ready")
        return file_path

    file_name,_ = urllib.request.urlretrieve(DOWNLOAD_URL + file_name,file_path)
    file_stat  = os.stat(file_path)
    if file_stat.st_size == expected_bytes:
        print("successfully downloaded the file",file_name)
    else:
        raise Exception("File" + file_name + "might be corrupted.")
    return file_path

def read_data(file_path):
    """ Read data into a list of tokens
    There should be 17,005,207 tokens
    """
    with zipfile.ZipFile(file_path) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return words

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def build_vocab(words,vocab_size):
    dictionary = dict()
    count = [("UNK",-1)]
    count .extend(Counter(words).most_common(vocab_size-1))
    index = 0
    make_dir('processed')
    with codecs.open('processed/vocab_1000.tsv',mode='w',encoding='utf-8') as f:
        for word,_ in count:
            dictionary[word] = index
            if index < 1000:
                f.write(word + "\n")
            index +=1
    index_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return dictionary,index_dictionary

def convert_words_to_index(words,dictionary):
    return [dictionary[word] if word in dictionary else 0 for word in words]

def generate_sample(index_words,context_window_size):
    """ Form training pairs according to the skip-gram model. """
    for index,center in enumerate(index_words):
        context = random.randint(1,context_window_size)
        # get a random target before the center word
        for target in index_words[max(0,index - context):index]:
            yield center,target
        # get a random target after the center word
        for target in index_words[index + 1:index + context + 1]:
            yield  center,target

def get_batch(iterator,batch_size):
    while True:
        center_batch = np.zeros(batch_size,dtype=np.int32)
        target_batch = np.zeros([batch_size,1])
        for index in range(batch_size):
            center_batch[index],target_batch[index] = next(iterator)
        yield center_batch,target_batch

def process_data(vocab_size,batch_size,skip_window):
    file_path = download(FILE_NAME,EXPECTED_BYTES)
    words = read_data(file_path)
    dictionary,_ = build_vocab(words,vocab_size)
    index_words = convert_words_to_index(words,dictionary)
    del words
    single_gen = generate_sample(index_words,skip_window)
    return get_batch(single_gen,batch_size)

def get_index_word(vocab_size):
    file_path = download(FILE_NAME,EXPECTED_BYTES)
    words = read_data(file_path)
    return build_vocab(words,vocab_size)
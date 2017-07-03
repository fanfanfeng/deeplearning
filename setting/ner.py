__author__ = 'fanfan'

from setting import defaultPath
import  os
data_path = os.path.join(defaultPath.PROJECT_DIRECTORY,'data/ner')
dev_path = os.path.join(data_path,'dev.txt')
test_path = os.path.join(data_path,'test.txt')
training_path = os.path.join(data_path,'train.txt')

word2vec_path = os.path.join(data_path,"vectors.model")
tag_to_id = {"O": 0, "B-LOC": 1, "I-LOC": 2,"B-PER": 3, "I-PER": 4, "B-ORG": 5, "I-ORG": 6}

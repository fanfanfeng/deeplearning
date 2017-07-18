import os
PROJECT_DIRECTORY ="E:\\python_work\\deeplearning" #os.path.dirname(os.path.abspath(os.curdir))


Stop_Words = r"data/stop_words.txt"

Mnist_data_path = os.path.join(PROJECT_DIRECTORY,r'data/mnist')

#2分类文本数据
Text_classfication_two = os.path.join(PROJECT_DIRECTORY,r'data/text_classfication_two')


#netura_networks_and deep_learning 数据path

netura_networks_mnist_path = os.path.join(PROJECT_DIRECTORY,r'data/netura_networks_and_deep_learning/mnist.pkl.gz')

#print(netura_networks_mnist_path)
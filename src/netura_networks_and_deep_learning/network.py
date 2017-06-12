__author__ = 'fanfan'
import random
import numpy as np
from src.netura_networks_and_deep_learning import mnist_loader

def sigmoid(z):
    return 1.0 /(1.0 + np.exp(-z))

def d_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Network():
    def __init__(self,sizes):
        """
        参数sizes代表层数以及个数
        [2,3,1]代表 一个3层神经网络，第一层2个神经元，第二层3个，第三层1个
        biases跟weights都用标准正态函数初始化
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

    def feedforward(self,a):
        """ Return the output of hte network  """
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self,traing_data,epochs,mini_batch_size,eta,test_data =None):
        if test_data:
            n_test = len(test_data)

        n = len(traing_data)
        for j in range(epochs):
            random.shuffle(traing_data)
            mini_batches = [
                traing_data[k:k + mini_batch_size] for k in range(0,n,mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)

            if test_data:
                print("Epoch {0}:{1} /{2}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {0} complete".format(j))


    def evaluate(self,data):
        test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in data]
        return sum(int(x == y) for (x,y) in test_results)

    def update_mini_batch(self,batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in batch:
            delta_nabla_b,delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw + dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]

            self.weights = [w - (eta/len(batch) * nw) for w,nw in zip(self.weights,nabla_w)]
            self.biases = [b - (eta/len(batch)*nb) for b,nb in zip(self.biases,nabla_b)]


    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []

        #feedfoward
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activation[-1],y) * d_sigmoid(zs[-1])
        nabla_b[-1] =  delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())

        for l in range(2,self.num_layers):
            z = zs[-l]
            sp = sigmoid(z)

            delta = np.dot(self.weights[-l+1].transpose(),delta) *sp

            nabla_b[-l]  = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())

        return (nabla_b,nabla_w)

    def cost_derivative(self,output_activations,y):
        return(output_activations - y)




if __name__ == '__main__':
    net = Network([2,3,1])
    print(net.biases[0].shape)
    print(net.biases[1].shape)

    print(net.weights[0].shape)
    print(net.weights[1].shape)

    a = np.array([[1.0],[2.0]])
    print(a.shape)
    a_forword = net.feedforward(a)
    print(a_forword)


    training_data,validation_data,test_data = mnist_loader.load_data_wrapper()
    net = Network([784,30,10])
    net.SGD(list(training_data),30,100,3.0,list(test_data))


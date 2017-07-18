# create by fanfan on 2017/7/3 0003
scores = [3.0,1.0,0.2]
import numpy as np

def softmax(x):
    y = []
    for i in x:
        y.append(np.exp(i))

    return y/sum(y)

print(softmax(scores))

import matplotlib.pyplot as plt
x = np.arange(-2.0,6.0,0.1)
scores = np.vstack([x,np.ones_like(x),0.2*np.ones_like(x)])
plt.plot(x,softmax(scores).T,linewidth=2)
plt.show()
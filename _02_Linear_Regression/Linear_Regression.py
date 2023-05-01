# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,Y=read_data()
    w=np.matmul(np.linalg.inv(np.matmul(X.T,X)),np.matmul(X.T,Y))
    return w @data
    
def lasso(data):
    item=10000
    min=0.0001
    alpha=1e-5
    X,Y=read_data()
    m,n = X.shape
    w2 = np.zeros(n)
    for i in range(item):
        grad=np.matmul(X.T,np.matmul(X,w2)-Y)+alpha*np.sign(w2)
        w2=w2-alpha*grad
        if np.linalg.norm(grad)<min:
            break
    return w2 @data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

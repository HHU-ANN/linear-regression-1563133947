# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,Y=read_data()
    X = np.mat(X)  # 404*6
    Y = np.mat(Y).T  # 404*1
    XTX=X.T*X
    rXTX=XTX+np.eye(X.shape[1])*0.0001
    new_w=rXTX.I*X.T*Y
    return new_w @ data
    
def lasso(data):
    item=1000
    alpha=0.0001
    tol=0.001
    X,Y=read_data()
    m,n = X.shape
    w = np.zeros(n)
    for i in range(item):
        grad=np.matmul(X.T,(np.matmul(X,w)-Y)+alpha*np.sign(w)
        w=w-alpha*grad
    return w @data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

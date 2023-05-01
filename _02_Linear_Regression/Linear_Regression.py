# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,Y=read_data()
    lamuda=0.01
    t1=np.linalg.inv((np.matmul(X.T,X)+lamuda*np.eye(X.shape[1])))
    w_r=np.matmul(t1,np.matmul(X.T,Y))
    return sum(w_r*data)
    
def lasso(data):
    X,Y=read_data() # 404*1
    alpha=1e-15
    step=1e-15
    item=10000
    tol=0.0001
    m,n= X.shape
    w = np.zeros(n)
    for i in range(item):
        Y_hat=np.dot(X,w)
        grad=np.dot(X.T,Y_hat-Y)/m+alpha*np.sign(w)
        w=w-step*grad
        if np.linalg.norm(grad)<tol:
            break
    return sum(w*data)

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

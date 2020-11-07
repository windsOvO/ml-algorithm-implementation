import numpy as np

def pca(X, k=None):
    '''
    X: sample matrix, shape(n, m)
    Y: principle component matrix, shape(n, k)

    n: number of samples
    m: number of original features
    k: number of principle components after PCA
    '''
    # normalization - make sum of column elements = 0
    X = np.array(X)
    # column => axis=0, row => axis=1
    X = (X - X.mean(axis=0))

    # construct matrix
    X_ = X / np.sqrt(X.shape[0] - 1) 

    # compacted SVD
    U, S, VT = np.linalg.svd(X_, full_matrices=False)
    # X_.shape(n, m)
    # VT.shape(r, m), r=ranke(X_), S.shape(r, r), U.shape(n, r)
    
    # truncating
    if k != None:
        VT = VT[0:k, :]

    # calculate principle component matrix
    # shape(k, m) * shape(m, n)
    Y = np.matmul(VT, X.T)

    return Y.T # shape(n, k)

def projectData(X, U, k):
    pass

'''
np.array() & np.matrix()
1. 两者都有.T操作以返回转置矩阵, 但是np.mat多了.H(共轭转置)和.I(逆矩阵)
2. np.array 可以表示超过1~n维的数据, 而np.mat只能用于二维
'''

'''
X = np.array([[1,4], [2,5] ])
# row mean
X = X - X.mean(axis=1, keepdims=True)
# [[2.5][3.5]]

# column mean
X = X - X.mean(axis=0)
# [1.5, 4.5] or [[1.5, 4.5]]
'''
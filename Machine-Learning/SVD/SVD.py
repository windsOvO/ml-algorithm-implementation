import numpy as np

'''
A: objective matrix of singular value decomposition, shape(m, n)

sigmas[i]: number i singular value of A (descending order)

U: left singular matrix: made of left singular vectors, shape(m, m) / (m, r) / (m, k)
Sigma: rectangular diagonal matrix made of singular values, shape(m, n) / (r, r) / (k, k)
V: right singular matrix: made of right singular vectors, shape(n, n) / (n, r) / (n, k)
'''

# calculate A: U * Sigma * V => A
def rebuildMatrix(U, Sigma, V):
    a = np.matmul(U, Sigma)
    a = np.matmul(a, np.transpose(V))
    return a

'''
eigenvalue: 特征值
eigenvector: 特征向量
'''
# sort by eigenvalue in descending order
def sortByEigenvalue(eigenvalues, eigenvectors):
    # sort eigenvalue by descending order(*-1), default ascending
    index = np.argsort(-1 * eigenvalues)
    eigenvalues = eigenvalues[index] # rearrange
    eigenvectors = eigenvectors[:, index]
    return eigenvalues, eigenvectors


def svd(A, full=True, k=None):
    '''
    full: full singular value decomposition or not
    k: number of preserved singular values
        if full==True -> full singular value decomposition - 完全奇异值分解
        if k=None -> k==r -> compact singular value decomposition - 紧奇异值分解
        if k!=None -> truncated sigular value decomposition - 截断奇异值分解
        0 < k < r, r=rank(A), 秩
    '''
    ## 1.calculate eigenvaluevalue and eigenvector of AT*A
    # AT * A
    AT_A = np.matmul(np.transpose(A), A)
    # eigenvalues(list) and corresponding normalized right eigenvectors(matrix, each column is a eigenvector)
    # right eigenvectors == right singular vectors
    eigenvalues, eigenvectors = np.linalg.eig(AT_A)
    eigenvalues, eigenvectors = sortByEigenvalue(eigenvalues, eigenvectors)

    if k == None:
        # 如果矩阵可以对角化，那么非0特征值的个数就等于矩阵的秩，其本身AT*A正定矩阵，特征值非负
        # 用比较代替不等于，对于float更加精准
        # number of singular values more than 0
        rankOfSigma = len(list(filter(lambda x: x > 0, eigenvalues)))
        k = rankOfSigma

    ## 2.calculate V - right singular vectors
    V = eigenvectors
    # clipping
    if full == False:
        V = V[:, 0:k]

    ## 3.calculate Sigma - diagonal matrix with singular values
    eigenvalues = np.array(eigenvalues)
    sigmas = np.sqrt(eigenvalues)
    Sigma = np.diag(sigmas)
    # clipping
    # drop singular value=0 corresponding vectors
    Sigma = Sigma[0:k, :]
    if full == False:
        Sigma = Sigma[0:k, 0:k]

    ## 4.calculate U
    U = np.zeros((A.shape[0], k)) # (m, k)

    if full == True:
        k == A.shape[1] # n
    for i in range(k):
        # eigenvector
        u = np.matmul(A, V[:, i])
        U[:, i] = np.transpose(u / sigmas[i])


    ## 5.get SVD
    return U, Sigma, V
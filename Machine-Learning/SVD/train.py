import SVD
import numpy as np

A = np.array([
    [1,0,0,0],
    [0,0,0,4],
    [0,3,0,0],
    [0,0,0,0],
    [2,0,0,0]
])

# full singular value decomposition
U, Sigma, V = SVD.svd(A, full=True)

# compact singular value decomposition
U, Sigma, V = SVD.svd(A, full=False)

# truncated singular value decomposition
U, Sigma, V = SVD.svd(A, full=False, k=2)

print('U:\n', U)

print('Sigma:\n', Sigma)

print('V:\n', V)

print('A:\n', SVD.rebuildMatrix(U, Sigma, V))
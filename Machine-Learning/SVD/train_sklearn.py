# only has truncated SVD
from sklearn.decomposition import TruncatedSVD
import numpy as np

A = np.array([
    [1,0,0,0],
    [0,0,0,4],
    [0,3,0,0],
    [0,0,0,0],
    [2,0,0,0]
])

# PCA要计算协方差矩阵,矩阵过大时计算资源不够,尝试截断SVD
svd = TruncatedSVD(2)
A = svd.fit_transform(A)

print(A)
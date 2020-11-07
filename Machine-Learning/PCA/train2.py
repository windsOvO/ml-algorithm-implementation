import numpy as np
import matplotlib.pyplot as plt
from PCA import pca

## example2
'''
generate linear data with Gaussian noise
'''
x1 = np.arange(-3, 3, 0.06)
x2 = 3 * x1 + 2
x2 += np.random.normal(0, 2, 100)
plt.scatter(x1, x2)
plt.show()

'''
数组合并method
np.append()
np.concatenate()
np.stack()
np.hstack(): 在水平方向上平铺
np.vstack(): 在竖直方向上堆叠
np.dstack(): 在深度方向进行拼接
'''
X = np.dstack((x1, x2))
X = np.squeeze(X, axis=0)
# print(X)

'''
training
'''

Y = pca(X, k=2)
# print(Y)
plt.clf()
plt.scatter(Y[:, 0], Y[:, 1])
plt.show()
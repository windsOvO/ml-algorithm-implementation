from PCA import pca
import numpy as np
import matplotlib.pyplot as plt

## example1
'''
loading data
'''
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

'''
training
k = number of preserved
'''
k = 2
x = pca(X, k)

'''
visualization
'''
plt.scatter(x[y == 0, 0], x[y == 0, 1], c="r", label=iris.target_names[0])
plt.scatter(x[y == 1, 0], x[y == 1, 1], c="b", label=iris.target_names[1])
plt.scatter(x[y == 2, 0], x[y == 2, 1], c="y", label=iris.target_names[2])
plt.legend()
plt.title("PCA of iris dataset")
plt.show()


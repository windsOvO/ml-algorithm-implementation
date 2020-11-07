from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

'''
loading data
X: unlabeled data, shape(n, m)
  m = pixel of photo, shape(8 * 8 = 64)
  n = number of samples, shape(1797)
y: labels of data
'''

digits = load_digits()
X = digits.data
y = digits.target

'''
training
'''

X_tsne = TSNE(n_components=2).fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X)

'''
visualization
compared with PCA
'''

plt.figure(figsize=(10, 5))
# t-sne
# plt.subplot(1,2,1), can be simplified as below
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y,label="t-SNE")
plt.legend()
# pca
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y,label="PCA")
plt.legend()
plt.show()
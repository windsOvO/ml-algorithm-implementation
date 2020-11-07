from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target

# n_components=2, 所要保留的成分个数
pca = PCA(2)
x = pca.fit_transform(x)

plt.scatter(x[y == 0, 0], x[y == 0, 1], c="r", label=iris.target_names[0])
plt.scatter(x[y == 1, 0], x[y == 1, 1], c="b", label=iris.target_names[1])
plt.scatter(x[y == 2, 0], x[y == 2, 1], c="y", label=iris.target_names[2])
plt.legend()
plt.title("PCA of iris dataset")
plt.show()

# 保存了0.95的信息
pca = PCA(n_components=0.80, svd_solver='full')  # 按信息保存率选维度
x = pca.fit_transform(x)
print(x.shape)
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import numpy as np
import umap

'''
loading data
'''
digits = load_digits()

'''
training
'''
reducer = umap.UMAP(random_state=51)
embedding = reducer.fit_transform(digits.data)
# print(embedding.shape) # (1797, 2)

'''
visualization
'''
plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
# gca == get current axes
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset')
plt.show()
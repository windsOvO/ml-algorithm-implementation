from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

'''
loading data and preprocessing
'''

# type: dict, data matrix from iris.data
iris = datasets.load_iris()

# 2-dimension plane
data = iris['data'][:, :2]

print(data)

'''
training and testing

inputs: data, shape(n, 2)
outputs:
    pointSet: dict(key:index of cluser, value: points list)
    centers: shape(k, 2)

n: number of examples
k: number of clusters
e: epoch of iteration
'''
from KMeans import KMeans

k = 3
e = 100

model = KMeans(k, e)
pointsSet, centers = model.fit(data)

## visualization
# plot centers
for i, p in enumerate(centers):
    plt.scatter(p[0], p[1], color='C{}'.format(i), marker='^',
                edgecolor='black', s=256)
# plot other points
c1 = np.asarray(pointsSet[0])
c2 = np.asarray(pointsSet[1])
c3 = np.asarray(pointsSet[2])

plt.scatter(c1[:,0], c1[:,1], color='green')
plt.scatter(c2[:,0], c2[:,1], color='red')
plt.scatter(c3[:,0], c3[:,1], color='blue')    
plt.title('KMeans clustering')
plt.xlim(4, 8)
plt.ylim(1, 5)
plt.show()




import numpy as np

class KMeans:
    def __init__(self, k, e=20):
        '''
        k: number of clusters
        e: epoch of iteration
        '''
        self.k = k
        self.e = e

    def fit(self, X, centers=None):
        '''
        X: feature matrix, shape(n, m)
        centers: assgin initial point, optional

        n: number of examples
        m: number of features

        pointsSet: set of points, type: dict(key:index of cluser, value: points list)

        '''
        # random choosing K points
        if centers == None:
            index = np.random.randint(low=0, high=len(X),
                                size=self.k) # k-dimension list
            centers = X[index] # k*m dimension matrix

        cnt = 0
        while cnt < self.e:

            # 1. create empty sets of every clusters
            pointsSet = {key: [] for key in range(self.k)}

            # 2. traverse all points and find their closest cluster center
            for point in X:
                # use squared Euclidean distance as d(xi, center)
                nearestIndex = np.argmin(
                    np.sum((centers - point) ** 2, axis=1))
                pointsSet[nearestIndex].append(point)

            # 3. compute new centers
            for i in range(self.k):
                centers[i] = np.sum(pointsSet[i], axis=0) / len(pointsSet[i])

            cnt += 1

        return pointsSet, centers
import numpy as np
data = [
    [28,29,28],
    [18,23,18],
    [11,22,16],
    [21,23,22],
    [26,29,26],
    [20,23,22],
    [16,22,22],
    [14,23,24],
    [24,29,24],
    [22,27,24]
]
data = np.array(data)

from KMeans import KMeans

k = 2
e = 100

model = KMeans(k, e)
pointsSet, centers = model.fit(data)

print(centers)

print(pointsSet[0])

print(pointsSet[1])

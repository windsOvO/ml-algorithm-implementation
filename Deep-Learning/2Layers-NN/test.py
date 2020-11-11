import numpy as np
import matplotlib.pyplot as plt

'''
data preparation
'''
from utils import *
X, Y = load_planar_dataset()

# Visualize the data
# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
# plt.show()

'''
training
'''
from nn import TwoLayerNN

# n_h = 4
# n_h = 6
# n_h = 8
n_h = 20

e = 10000
# e = 50000

model = TwoLayerNN(X, Y, n_h)
model.train(epoch=e, print_cost=True)


'''
visualization
'''

plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
plt.show()

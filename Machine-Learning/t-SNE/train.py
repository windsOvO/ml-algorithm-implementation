from TSNE import *
import pylab

'''
loading MNIST dataset, training and visualization
'''

print("Running example on 2,500 MNIST digits...")
X = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")
Y = tsne(X, 2, 20.0, 1000)
pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
pylab.show()



'''
loading sklearn.digits data
X: unlabeled data, shape(n, m)
  m = pixel of photo, shape(8 * 8 = 64)
  n = number of samples, shape(1797)
y: labels of data
'''
# from sklearn.datasets import load_digits
# digits = load_digits()
# X = digits.data
# y = digits.target

# '''
# training
# '''
# Y = tsne(X, 2, 20.0)

# '''
# visualization
# '''
# from matplotlib import pyplot as plt
# plt.scatter(Y[:, 0], Y[:, 1], c=y,label="t-SNE")
# plt.legend()
# plt.show()
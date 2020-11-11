'''
uncompleted
'''


'''
data preparation
'''
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# X_train, shape[1347, 64]
# y_train, shape[1347]

# Y: shape[1, m]
# convenience calculate and normalize
X_train = X_train.T / 16
y_train = y_train.T.reshape(1, -1)

# print(X_train)

'''
training
'''

from dnn import DNN

# network structure
layers_dims = [12288, 20, 7, 5, 1]

model = DNN(layers_dims)

# need change last layer to softmax
model.train(X_train, y_train, lr=1.3, epoch=1000, print_cost=True)




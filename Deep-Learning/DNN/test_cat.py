import numpy as np
import matplotlib.pyplot as plt

'''
data preparation
'''
from utils import *
train_x_orig, y_train, test_x_orig, y_test, classes = load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

X_train = train_x_flatten/255.
X_test = test_x_flatten/255.

# train_x's shape: (12288, 209)
# test_x's shape: (12288, 50)

'''
training
'''
from dnn import DNN

# network structure
layers_dims = [12288, 20, 7, 5, 1]
# layers_dims = [12288, 7, 1]
# layers_dims = [12288, 1]

model = DNN(layers_dims)

model.train(X_train, y_train, lr=0.0075, epoch=5000, print_cost=True)

model.predict(X_test, y_test)


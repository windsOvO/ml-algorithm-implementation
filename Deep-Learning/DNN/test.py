import numpy as np

class TwoLayerNN:

    def __init__(self, X, Y, n_h=4):
        '''
        X: features, shape(n_x. m)
        Y: labels, shape(n_y, m)
            unconventional dimension define use to convenience calculation
        m: number of samples
        n_x: dimension of input
        n_h: dimension of 1st hidden layer
        n_h: dimension of output
        parameters: parameters after training
        '''
        self.X = X
        self.Y = Y
        self.n_h = n_h
        self.parameters = None


    def init_parameters(self, n_x, n_h, n_y):
        '''
        W1, b1, W2, b2: parameters in matrix operation
            X * W1 + b1 = Z
            A * W1 + b2 = Y
        '''
        np.random.seed(2)

        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((1,1))

        parameters = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2
        }

        return parameters


    def forward_propagation(self, X, parameters):
        '''
        Z1 = W1 * X + b1, output of 1st matrix calculation
        A1 = tanh(Z1), 1st activated output
        Z2 = W2 * A1 + b2, output of 2nd matrix calculation
        A2 = sigmoid(Z2), 2st activated output
            b1 and b2 use numpy broadcast
        cache: the cache store updated parameters, and will be used in backprop
        '''
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]   
        
        Z1 = np.matmul(W1, X) + b1
        A1 = np.tanh(Z1) # choose tanh nonlinear function
        Z2 = np.matmul(W2, A1) + b2
        A2 = self.sigmoid(Z2) # choose sigmoid nonlinear function

        cache = {
            'Z1': Z1,
            'A1': A1,
            'Z2': Z2,
            'A2': A2
        }

        return A2, cache


    def compute_cost(self, A2, Y):
        '''
        m: number of samples
        '''
        m = Y.shape[1]
        # use cross entropy as loss function
        cross_entropy = Y * np.log(A2) + (1 - Y) * np.log(1 - A2)
        cost = -np.sum(cross_entropy) / m
        # formalize type
        cost = float(np.squeeze(cost))

        return cost


    def backward_propagation(self, parameters, cache, X, Y):
        '''
        m: number of samples
        dXx: gradient of Xx in backward propagation
        '''
        m = X.shape[1]
        W1 = parameters['W1']
        W2 = parameters['W2']
        A1 = cache['A1']
        A2 = cache['A2']

        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2)) # A1 * (1 - A1) activate function are different
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        
        grads = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
        
        return grads


    def update_parameters(self, parameters, grads, lr = 1.2):
        '''
        lr: learning_rate
        '''
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']
        # update
        W1 = W1 - lr * dW1
        b1 = b1 - lr * db1
        W2 = W2 - lr * dW2
        b2 = b2 - lr * db2

        parameters = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2
        }
    
        return parameters


    def train(self, epoch, print_cost=False):
        # initialize
        n_x = self.X.shape[0]
        n_y = self.Y.shape[0]
        parameters = self.init_parameters(n_x, self.n_h, n_y)

        # train in loop
        for i in range(0, epoch):
            # forward propagation
            A2, cache = self.forward_propagation(self.X, parameters)
            # compute cost
            cost = self.compute_cost(A2, self.Y)
            # backward propagation
            grads = self.backward_propagation(parameters, cache, self.X, self.Y)
            # update parameters
            parameters = self.update_parameters(parameters, grads)
        
            # print cost
            if print_cost and i % 1000 == 0:
                print('[%d/%d] cost: %f' % (i, epoch, cost))

        # store updated parameters
        self.parameters = parameters
        return

    
    def predict(self, X):
        A2, _ = self.forward_propagation(X, self.parameters)
        # classifies to 0/1 using 0.5 as the threshold
        predictions = np.round(A2)
        return predictions


    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A

from utils import *
train_x_orig, y_train, test_x_orig, y_test, classes = load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

X_train = train_x_flatten/255.
X_test = test_x_flatten/255.


model = TwoLayerNN(X, Y, n_h)
model.train(epoch=e, print_cost=True)

import numpy as np
import matplotlib.pyplot as plt

class DNN:

    def __init__(self, layers_dims):
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
        self.layers_dims = layers_dims
        self.parameters = None


    def init_parameters(self, layers_dims):
        '''
        layers_dims: list, every dimensions of each layer, 
                    including output layer and input layer(layers_dims[0]).
                    hidden layer do not include input layer
        length: length of layer
        '''
        np.random.seed(1)
        parameters = {}
        length = len(layers_dims)

        for l in range(1, length):
            # this initialization is corresponding to test_cat
            # if W too small, weight of b will increase, so every different input will have same output
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) \
                                            / np.sqrt(layers_dims[l-1]) # * 0.01
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        return parameters
    

    # calculate matrix multiply
    def linear_forward(self, A, W, b):
        '''
        cache: store median value for computing gradient efficiently
        '''
        Z = np.matmul(W, A) + b
        cache = (A, W, b)
        return Z, cache

    # calculate matrix multiply and nonlinear activation
    def linear_activation_forward(self, A_prev, W, b, activation):
        '''
        cache: store median value for computing gradient efficiently
        '''
        if activation == 'sigmoid':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
        elif activation == 'relu':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
        
        cache = (linear_cache, activation_cache)
        return A, cache

    def forward_propagation(self, X, parameters):
        '''
        caches: the cache store updated parameters, and will be used in backprop
        '''
        caches = []
        A = X # origin activation value
        length = len(parameters) // 2 # [W1, b1...]
        # // -> exact division
        
        # midden layers use relu
        for l in range(1, length):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev,
                    parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
            caches.append(cache)
        
        # last layer use sigmoid
        AL, cache = self.linear_activation_forward(A,
                    parameters['W' + str(length)], parameters['b' + str(length)], 'sigmoid')
        caches.append(cache)

        return AL, caches


    def compute_cost(self, AL, Y):
        '''
        m: number of samples
        AL, Y: shape[1, m]
        '''
        m = Y.shape[1]
        # use cross entropy as loss function
        cross_entropy = Y * np.log(AL) + (1 - Y) * np.log(1 - AL)
        cost = -np.sum(cross_entropy) / m
        # formalize type
        cost = float(np.squeeze(cost))

        return cost

    # calculate gradient in matrix multiply
    def linear_backward(self, dZ, cache):
        '''
        m: number of samples
        dX: gradient of Xx in backward propagation
        '''
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.matmul(W.T, dZ)

        return dA_prev, dW, db

    # calculate gradient in matrix multiply and activation function
    def linear_activation_backward(self, dA, cache, activation):
        """
        dX: gradient of Xx in backward propagation
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        
        return dA_prev, dW, db


    def backward_propagation(self, AL, Y, caches):
        '''
        m: number of samples
        grads: dict, store all gradients
        '''
        grads = {}
        length = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # formalize

        # numpy.divde -> divide corresponding position
        # AL gradient in cross entropy
        dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))
        # print(AL[0, 1], Y[0, 1], dAL[0, 1])
        # print(AL[0, 2], Y[0, 2], dAL[0, 2])

        # last layer use sigmoid
        current_cache = caches[length - 1]
        grads["dA" + str(length-1)], grads["dW" + str(length)], grads["db" + str(length)] = \
                        self.linear_activation_backward(dAL, current_cache, 'sigmoid')

        # midden layers use relu
        # loop from l=L-2 to l=0
        for l in reversed(range(length - 1)):
            current_cache = caches[l]
            dA_prev, dW, db = \
                    self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, 'relu')
            grads["dA" + str(l)] = dA_prev
            grads["dW" + str(l + 1)] = dW
            grads["db" + str(l + 1)] = db
        
        return grads


    def update_parameters(self, parameters, grads, lr):
        '''
        lr: learning_rate
        '''
        length = len(parameters) // 2
        # update
        for l in range(length):
            parameters["W" + str(l+1)] = \
                parameters["W" + str(l + 1)] - lr * grads["dW" + str(l + 1)]
            parameters["b" + str(l+1)] = \
                parameters["b" + str(l + 1)] - lr * grads["db" + str(l + 1)]
    
        return parameters


    def train(self, X, Y, lr = 0.0075, epoch = 3000, print_cost=False):
        '''
        lr: learning rate
        epoch: number of iterations
        cost: list, keep track of cost
        '''
        # initialize
        np.random.seed(1)
        costs = []
        parameters = self.init_parameters(self.layers_dims)

        # train in loop
        for i in range(0, epoch):
            # forward propagation
            AL, caches = self.forward_propagation(X, parameters)
            # compute cost
            cost = self.compute_cost(AL, Y)
            # backward propagation
            grads = self.backward_propagation(AL, Y, caches)
            # update parameters
            parameters = self.update_parameters(parameters, grads, lr)
        
            # print cost
            if print_cost and i % 100 == 0:
                print('[%d/%d] cost: %f' % (i, epoch, cost))
                costs.append(cost)
        
        # visualize costs
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(lr))
        plt.show()

        # store updated parameters
        self.parameters = parameters
        return

    
    def predict(self, X, Y):
        m = X.shape[1]

        AL, _ = self.forward_propagation(X, self.parameters)
        # classifies to 0/1 using 0.5 as the threshold
        predictions = np.round(AL)

        print("Accuracy: "  + str(np.sum((predictions == Y)/m)))
        
        return predictions

    # activation function
    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        cache = Z
        return A, cache

    def relu(self, Z):
        A = np.maximum(0, Z)
        cache = Z
        return A, cache
    
    # gradient in activation function
    def sigmoid_backward(self, dA, cache):
        Z = cache
        s = 1 / (1 + np.exp(-Z)) # sigmoid
        dZ = dA * s * (1-s) # nature of sigmoid
        return dZ

    def relu_backward(self, dA, cache):
        Z = cache
        # must copy new object
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

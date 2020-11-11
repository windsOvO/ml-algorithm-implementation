import numpy as np

'''
Classfication Tree
'''
class ClassficationTree:
    def __init__(self, X, y, epsilon):
        '''
        X: sample matrix
        y: labels
        epsilon: threshold, stop when sample numbers less than epsilon
        num_feature: number of features
        tree: the root node of tree
        '''
        self.X = X
        self.y = y
        self.num_feature = X.shape[1]
        self.epsilon = epsilon
        self.tree = None


    def _fit(self, X, y, epsilon):
        # get best splitting variable and splitting point
        (j, s, min_cost, c1, c2) = self._divide(X, y, self.num_feature)
        # initialize tree
        tree = {"feature": j, "value": X[s, j], "left": None, "right": None}

        value = X[s, j] # splitting point

        #  lower than threshold or cannot be divide
        if min_cost < self.epsilon or len(y[np.where(X[:, j] <= value)]) <= 1:
            tree["left"] = {"feature": None, "value": c1, "left": None, "right": None}
        # recurse
        else:
            tree["left"] = self._fit(X[np.where(X[:, j] <= value)],
                                     y[np.where(X[:, j] <= value)],
                                     self.epsilon)
        
        if min_cost < self.epsilon or len(y[np.where(X[:, j] > value)]) <= 1:
            tree["right"] = {"feature": None, "value": c2, "left": None, "right": None}
        else:
            tree["right"] = self._fit(X[np.where(X[:, j] > value)],
                                      y[np.where(X[:, j] > value)],
                                      self.epsilon)
        return tree


    def fit(self):
        self.tree = self._fit(self.X, self.y, self.epsilon)


    ## divide input space to find best splitting variable and splitting point
    def _divide(self, X, y, num_feature):
        '''
        num_feature: number of feature, also known as m
        cost: cost matrix
        cost_index: index of minimum cost
        '''
        # initialize cost
        cost = np.zeros((num_feature, n))
        # find best splitting feature
        for i in range(num_feature):
            value = X[k, i] # splitting point
                # left
                y1 = y[np.where(X[:, i] <= value)]
                c1 = np.mean(y1)
                # right
                y2 = y[np.where(X[:, i] > value)]
                # avoid mean empty bug
                if y2 != []:
                    c2 = np.mean(y2)
                else:
                    y2 = c2 = 0
                # calculate mse cost
                cost[i, k] = np.sum(np.square(y1 - c1)) + \
                                np.sum(np.square(y2 - c2))
                
        
        # find minimum cost
        cost_index = np.where(cost == np.min(cost))

        j = cost_index[0][0] # splitting variavle
        s = cost_index[1][0] # splitting point
        # recalculate mse cost
        value = X[s, j]
        c1 = np.mean(y[np.where(X[:, j] <= value)])
        c2 = np.mean(y[np.where(X[:, j] > value)])

        return j, s, cost[cost_index], c1, c2

    def predict(self, x):
        return self._predict(x, self.tree)


    def _predict(self, x, tree):
        '''
        x: one input
        f(x) = sigma[M][m=1](c * I(x belong to Rm))
        c: objective ouput
        '''
        while True:
            feature, c, left, right = tree.values()
            # I(x belong to Rm)
            if left == None and right == None:
                return c
            else:
                value = x[feature]
                if value <= c:
                    return self._predict(x, left)
                elif value > c:
                    return self._predict(x, right)



'''
Regression Tree
'''
class RegressionTree:
    def __init__(self, X, y, epsilon):
        '''
        X: sample matrix
        y: labels
        epsilon: threshold of MSE
        num_feature: number of features
        tree: the root node of tree
        '''
        self.X = X
        self.y = y
        self.num_feature = X.shape[1]
        self.epsilon = epsilon
        self.tree = None


    # _function -> private methods
    def _fit(self, X, y, epsilon):
        # get best splitting variable and splitting point
        (j, s, min_cost, c1, c2) = self._divide(X, y, self.num_feature)
        # initialize tree
        tree = {"feature": j, "value": X[s, j], "left": None, "right": None}

        value = X[s, j] # splitting point

        #  lower than threshold or cannot be divide
        if min_cost < self.epsilon or len(y[np.where(X[:, j] <= value)]) <= 1:
            tree["left"] = {"feature": None, "value": c1, "left": None, "right": None}
        # recurse
        else:
            tree["left"] = self._fit(X[np.where(X[:, j] <= value)],
                                     y[np.where(X[:, j] <= value)],
                                     self.epsilon)
        
        if min_cost < self.epsilon or len(y[np.where(X[:, j] > value)]) <= 1:
            tree["right"] = {"feature": None, "value": c2, "left": None, "right": None}
        else:
            tree["right"] = self._fit(X[np.where(X[:, j] > value)],
                                      y[np.where(X[:, j] > value)],
                                      self.epsilon)
        return tree


    def fit(self):
        self.tree = self._fit(self.X, self.y, self.epsilon)


    ## divide input space to find best splitting variable and splitting point
    def _divide(self, X, y, num_feature):
        '''
        num_feature: number of feature, also known as m
        cost: cost matrix
        cost_index: index of minimum cost
        '''
        # initialize cost
        n = X.shape[0]
        cost = np.zeros((num_feature, n))
        # find best splitting variable
        for i in range(num_feature):
            # find best splitting point
            for k in range(n):
                value = X[k, i] # splitting point
                # left
                y1 = y[np.where(X[:, i] <= value)]
                c1 = np.mean(y1)
                # right
                y2 = y[np.where(X[:, i] > value)]
                # avoid mean empty bug
                if y2 != []:
                    c2 = np.mean(y2)
                else:
                    y2 = c2 = 0
                # calculate mse cost
                cost[i, k] = np.sum(np.square(y1 - c1)) + \
                                np.sum(np.square(y2 - c2))
        
        # find minimum cost
        cost_index = np.where(cost == np.min(cost))

        j = cost_index[0][0] # splitting variavle
        s = cost_index[1][0] # splitting point
        # recalculate mse cost
        value = X[s, j]
        c1 = np.mean(y[np.where(X[:, j] <= value)])
        c2 = np.mean(y[np.where(X[:, j] > value)])

        return j, s, cost[cost_index], c1, c2

    def predict(self, x):
        return self._predict(x, self.tree)


    def _predict(self, x, tree):
        '''
        x: one input
        f(x) = sigma[M][m=1](c * I(x belong to Rm))
        c: objective ouput
        '''
        while True:
            feature, c, left, right = tree.values()
            # I(x belong to Rm)
            if left == None and right == None:
                return c
            else:
                value = x[feature]
                if value <= c:
                    return self._predict(x, left)
                elif value > c:
                    return self._predict(x, right)

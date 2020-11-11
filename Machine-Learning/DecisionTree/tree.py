import numpy as np

# X, y are both numpy array

class Tree:

    def __init__():

    def calc_entropy(y):
        '''
        y: labels of each samples in a list
        n: number of samples, also known as |D|
        label_set: all unique labels of samples
        num_Ci: number of specific labels, also known as |Ci|
        pi: probability of specific label in dataset
        '''
        entropy = 0
        n = y.size # .size: a number equal to shape[0]*...*shape[k-1]
        label_set = set(y.tolist())
        # traverse all labels
        for label in label_set:
            # use numpy broadcast to calculate |Ci|
            # this method can be seem as condition distribution
            num_Ci = y[y == label].size
            # calculate pi = |Ci| / |D|
            pi = num_Ci / n
            # H(X) = H(p) = -sigma[n][i=1](pi * log(pi)
            entropy += -1 * p * np.log2(p)
        
        return entropy


    def calc_feature_entropy(y):
        '''
        y: labels of each samples in a list
        n: number of samples, also known as |D|
        label_set: all unique labels of samples
        num_Ci: number of specific labels, also known as |Ci|
        pi: probability of specific label in dataset
        '''
        entropy = 0
        n = y.size # .size: a number equal to shape[0]*...*shape[k-1]
        label_set = set(y.tolist())
        # traverse all labels
        for label in label_set:
            # use numpy broadcast to calculate |Ci|
            # this method can be seem as condition distribution
            num_Ci = y[y == label].size
            # calculate pi = |Ci| / |D|
            pi = num_Ci / n
            # H(X) = H(p) = -sigma[n][i=1](pi * log(pi)
            entropy += -1 * p * np.log2(p)
        
        return entropy

    def calc_conditon_entropy(x_feature, y):
        '''
        x_feature: data in one feature, x = X[][i]
        y: labels of each samples in a list
        n: number of samples, also known as |D|
        value_set: all unique values of one feature
        pi: probability of specific value in one feature
        '''
        n = x_feature.size
        value_set = set(x_feature.tolist()) 
        for value in value_set:
            num_Di = x[x == value].size
            pi = num_Di / n
            # H(Y|X) = sigma[n][i=1](pi * H(Y|X=xi))
            condition_entropy += pi * calc_entropy(y[x == i])

        return conditon_entropy


    def info_gain(entropy, condition_entropy):
        return entropy - condition_entropy

    def info_gain_ratio(entropy, condition_entropy):
        return condition_entropy / entropy


    def calc_best_feature(X, y, mode='ID3'):
        '''
        
        '''
        num_feature = X.shape[1]

        max_info_gain = None
        max_info_gain_ratio = None
        best_feature = None

        for feature in range(num_feature):
            if mode == 'ID3':

            elif mode == 'C4.5':

        return best_feature, gain



    def createTree(X, y, epsilon):
        epsilon = 0.1

        feature, gain = calc_best_feature(X, y)
        if gain < epsilon:
            return 

    def predict(X):

        while True:
            pass
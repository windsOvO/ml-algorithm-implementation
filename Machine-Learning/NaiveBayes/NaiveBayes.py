import numpy as np
import math

class NaiveBayes:
    def __init__(self):
        '''
        unique_labels: all probable labels genre

        Py: distribution of prior probability, shape(k)
        Px_y: distribution of conditional probability, shape(k, m, l)
        '''
        self.unqiue_labels = None
        self.Py = None
        self.Px_y = None


    def fit(self, X, y, lambda0=0):
        '''
        X: feature matrix, shape(n, m), numpy
        y: labels vector, shape(n), numpy
        lambda0: parameter of Bayes estimation, lambda=0->MLE

        unique_feature: all probable feature genre
        unique_labels: all probable labels genre

        n: number of examples
        m: number of features
        k: number of labels genres
        l: number of feature genres

        Pxy:  distribution of joint probability without normalization, shape(k, m, l)
        '''
        assert X.shape[0] == y.shape[0]

        ## calculate shape
        n = X.shape[0]
        m = X.shape[1]

        # calculate feature genres
        x_list = []
        for x in X.tolist():
            x_list.extend(x)
        unique_features = list(set(x_list))
        unique_labels = list(set(y))
        k = len(unique_labels)
        l = len(unique_features)

        ## calculate probability
        Py = np.zeros(k)
        for i, label in enumerate(unique_labels):
            # compare every label with i
            denominator = np.sum(y == label) + lambda0 # use broadcast in numpy
            numerator = m + k * lambda0
            Py[i] = denominator / numerator
        # Py = np.log(Py) # log MLE

        # joint probability
        Pxy = np.zeros((k, m, l))
        for i in range(n):
            label = y[i] 
            x = X[i]
            for j in range(m):
                Pxy[label][j][x[j]] += 1

        # conditional probability
        Px_y = np.zeros((k, m, l))
        for i in range(k):
            for j in range(m):
                for h in range(l):
                    denominator = Pxy[label][j][h] + lambda0 # use broadcast in numpy
                    numerator = np.sum(Px_y, axis=2) + l * lambda0
                    Px_y[label][j][0] = denominator / numerator

        print('Model train done!')
        
        def predict(self, x):
            '''
            P: posterior probability， shape(k)
            '''
            k = Px_y.shape[0]
            m = Px_y.shape[1]

            P = np.zeros(k)

            for i in range(k):
                product = 1
                for j in range(m):
                    product *= Px_y[i][j][x[j]]
                P[i] = Py[i] * product
            
            return unique_labels[np.argmax(P)]
        
        def score(self, X_test, y_test):
            right = 0
            for x, y in zip(X_test, y_test):
                label = self.predict(x)
                if label == y:
                    right += 1
            return right / float(X_test.shape[0])
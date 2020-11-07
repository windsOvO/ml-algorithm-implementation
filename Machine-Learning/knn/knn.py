import numpy as np
import operator 

class KNN:

    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        # knn无训练过程
        self.X = X
        self.y = y

    # __私有函数（双下划线）
    def __square_distance(self, v1, v2):
        return np.sum(np.square(v1 - v2))

    def __vote(self, y):
        # 求出互异的值
        y_unique = np.unique(y)
        vote_dict = {}
        for i in y:
            if i not in vote_dict.keys():
                vote_dict[i] = 1
            else:
                vote_dict[i] += 1
        # sorted: python内置排序，与sort不同，其可以在所有迭代对象进行操作
        # operator.itemgetter(1)，获取对象(子数组)的第1个值
        sorted_vote_dict = sorted(vote_dict.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_vote_dict[0][0]

    #
    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            dist_arr = [self.__square_distance(X[i], self.X[j]) for j in range(len(self.X))]
            # np.argsort: 返回排序后的坐标
            sorted_index = np.argsort(dist_arr)
            top_k_index = sorted_index[:self.k]
            y_pred.append(self.__vote(y=self.y[top_k_index]))
        return np.array(y_pred)

    def score(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_pred = self.predict(self.X)
            y_true = self.y
        score = 0.0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                score += 1
        score /=  len(y_true)
        return score
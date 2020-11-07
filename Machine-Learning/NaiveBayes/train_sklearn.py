from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

'''
loading data and proprocessing
'''

def createData():
    iris = load_iris()
    # use pa to load data from matrix
    df = pd.DataFrame(iris.data)
    # add labels to dataframe
    df['label'] = iris.target
    # rename features
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]

    # loc: locate and get line with name their name
    # iloc: locate and get line with name their index
    data = np.array(df.iloc[:, :])

    # shuffle data
    data = shuffle(data)

    # just get the first 100 data
    data = data[:, :]
    # print(data)

    return data[:, :-1], data[:, -1]

X, y = createData()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

'''
training and testing
X: shape(n, 4)
y: one in [0, 1, 2]
'''

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

model = GaussianNB()
model.fit(X_train, y_train)

pred = model.predict([[4.4,  3.2,  1.3,  0.2]])
score = model.score(X_test, y_test)

print(pred)
print(score)
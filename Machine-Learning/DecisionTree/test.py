from CART import *

X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T
y = np.array([4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00])

model = RegressionTree(X, y, 0.2)
model.fit()
# print(model.tree)

'''
{'feature': 0,
 'value': 5,
 'left': {'feature': 0, 'value': 3, 'left': 4.72, 'right': 5.57},
 'right': {'feature': 0,
  'value': 7,
  'left': {'feature': 0, 'value': 6, 'left': 7.05, 'right': 7.9},
  'right': {'feature': 0, 'value': 8, 'left': 8.23, 'right': 8.85}}}
...

f(x) = 4.72 x<=3
       5.57 3<x<=5
       7.05 5<x<=6
       7.9  6<x<=7
       8.23 7<x<=8
       8.85 x>8

'''

X_test = np.array([7.5])
predictions = model.predict(X_test)

print(predictions)
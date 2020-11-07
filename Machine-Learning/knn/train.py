import numpy as np
import matplotlib.pyplot as plt
import knn


### data generation
np.random.seed(272)
data_size_1 = 300
# normal argu: loc=中心值，scale=宽度
x1_1 = np.random.normal(loc=5.0, scale=1.0, size=data_size_1)
x2_1 = np.random.normal(loc=4.0, scale=1.0, size=data_size_1)
# list = [f(i) for i in dataSet] - 方便创建列表
# 将循环产生的i结果（dataSet中数据），通过f函数映射，放入列表list中
y_1 = [0 for _ in range(data_size_1)]

data_size_2 = 400
x1_2 = np.random.normal(loc=10.0, scale=2.0, size=data_size_2)
x2_2 = np.random.normal(loc=8.0, scale=2.0, size=data_size_2)
y_2 = [1 for _ in range(data_size_2)]


### data disposal
# data concatenate
# concatenate: v./adj.连接，np中：数组连接
x1 = np.concatenate((x1_1, x1_2), axis=0)
x2 = np.concatenate((x2_1, x2_2), axis=0)
# np.hstack(): 水平方向
# reshape一个值为-1，就表示待定，先确保其他纬度，然后确定-1的值
x = np.hstack((x1.reshape(-1, 1), x2.reshape(-1,1)))
y = np.concatenate((y_1, y_2), axis=0)

# data shuffle
data_size_all = data_size_1 + data_size_2
# shuffle: 洗牌，permutation: 排列；置换
shuffled_index = np.random.permutation(data_size_all) # 生成打乱后的index list
x = x[shuffled_index]
y = y[shuffled_index]

# data split - 73分训练集和测试集
split_index = int(data_size_all * 0.7)
x_train = x[:split_index]
y_train = y[:split_index]
x_test = x[split_index:]
y_test = y[split_index:]


### data visualization
# c: color, marker: 散点形状
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, marker='.')
# plt.show()
plt.scatter(x_test[:,0], x_test[:,1], c=y_test, marker='.')
# plt.show()


### data preprocessing
# normalization
x_train = (x_train - np.min(x_train, axis=0)) / (np.max(x_train, axis=0) - np.min(x_train, axis=0))
x_test = (x_test - np.min(x_test, axis=0)) / (np.max(x_test, axis=0) - np.min(x_test, axis=0))


### knn classifier
clf = knn.KNN(k=3)
clf.fit(x_train, y_train)
score_train = clf.score()

print('train accuracy: {:.3}'.format(score_train))

y_test_pred = clf.predict(x_test)
print('test accuracy: {:.3}'.format(clf.score(y_test, y_test_pred)))
# encoding=utf-8
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import LeaveOneOut, LeavePOut

# 数据集划分一：简单交叉验证
X = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]
y = [0, 0, 0, 0, 1, 1, 1, 1]
# 未指定分层采样
Z = train_test_split(X, y, test_size=0.4, random_state=3)
(X_train, X_test, y_train, y_test) = Z
print("未指定分层采样:", y_train, y_test)
# 根据y分层采样
Z = train_test_split(X, y, test_size=0.4, random_state=3, stratify=y)
(X_Train, X_Test, y_train, y_test) = Z
print("已指定分层采样:", y_train, y_test, "\n")

# 数据集划分二：k折交叉验证
X = np.array([[0, 1], [1, 2], [2, 3], [3, 4],
              [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
kf = KFold(n_splits=3)  # 未随机打乱数据
print("3折交叉验证(未随机打乱数据)：")
for train_index, test_index in kf.split(X, y):
    print(train_index, test_index)
    print(y[train_index], y[test_index])
print("3折交叉验证(随机打乱数据)：")
kf = KFold(n_splits=3, shuffle=True)  # 随机打乱数据
for train_index, test_index in kf.split(X, y):
    print(train_index, test_index)
    print(y[train_index], y[test_index])
print("分层采样4折交叉验证:")
kf = StratifiedKFold(n_splits=4, shuffle=True)  # 分层采样k折交叉验证
for train_index, test_index in kf.split(X, y):
    print(train_index, test_index)
    print(y[train_index], y[test_index])

# 数据集划分三：留一法
X = [[0, 1], [1, 2], [2, 3], [3, 4]]
y = [0, 0, 1, 1]
lo = LeaveOneOut()  # 留一法
print("\n留一法:")
for train_index, test_index in lo.split(X, y):
    print(train_index, test_index)
print("留P法(p=2)")
lpo = LeavePOut(2)  # 留P法
for train_index, test_index in lpo.split(X, y):
    print(train_index, test_index)

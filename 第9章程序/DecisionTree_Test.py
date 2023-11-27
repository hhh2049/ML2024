# encoding=utf-8
import numpy as np
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree

def main():
    X = np.array([[0, 1, 1, 1, 0],
                  [0, 1, 2, 1, 0],
                  [1, 1, 0, 1, 1],
                  [0, 0, 2, 0, 0],
                  [0, 1, 0, 0, 2],
                  [0, 1, 2, 0, 0],
                  [0, 1, 1, 1, 2],
                  [2, 1, 2, 1, 1],
                  [2, 0, 1, 0, 0],
                  [0, 1, 0, 1, 2],
                  [2, 1, 2, 1, 0],
                  [1, 1, 2, 1, 1]])                            # 小丽相亲训练数据集
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])         # 训练数据的分类标签

    Z = train_test_split(X, y, test_size=0.3, random_state=1)  # 划分训练集和测试集
    (X_train, X_test, y_train, y_test) = Z                     # 训练集和测试集赋值

    our_dt = DecisionTree(X_train, y_train)                    # 定义自实现决策树对象
    our_dt.fit()                                               # 训练模型

    train_score = our_dt.score(X_train, y_train)               # 计算训练得分
    test_score  = our_dt.score(X_test, y_test)                 # 计算预测得分
    print("train score = %f" % train_score)                    # 打印训练得分
    print("test  score = %f" % test_score)                     # 打印预测得分

if __name__ == "__main__":
    main()

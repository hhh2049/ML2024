# encoding=utf-8
import numpy as np
from CART_Weight import CART

class AdaBoostClassifier:  # Adaboost分类的实现
    def __init__(self, X, y, n_estimators=10):
        self.X            = X                                  # 训练数据集
        self.y            = y                                  # 训练数据的分类标签
        self.n_estimators = n_estimators                       # 弱学习器的数量
        self.e            = np.zeros(n_estimators)             # 各弱学习器的错误率
        self.a            = np.zeros(n_estimators)             # 各弱学习器的权重
        self.tree         = []                                 # 存放弱学习器的列表

    def fit(self):  # 拟合数据，训练模型
        X, y = self.X, self.y                                  # 为缩短代码长度
        m = X.shape[0]                                         # 训练数据集的样本量
        wt = np.array([1/m] * m)                               # 初始各个样本的权重

        for t in range(self.n_estimators):                     # 尝试生成各个弱学习器
            tree = CART(X, y, sample_weight=wt)                # 弱学习器，带权重，深度1
            tree.fit()                                         # 训练弱学习器

            self.e[t] = 1 - tree.score(X, y, wt)               # 计算当前弱学习器错误率
            if self.e[t] > 0.5:                                # 如果错误率超过0.5
                print("train wrong e > 0.5")                   # 打印错误信息
                break                                          # 中止训练
            self.a[t] = 0.5 * np.log((1-self.e[t])/self.e[t])  # 计算当前弱学习器的权重

            for i in range(m):                                 # 遍历所有数据
                yi_ = tree.predict_one(X[i])                   # 计算当前弱学习器预测值
                wt[i] *= np.exp(-self.a[t] * y[i] * yi_)       # 更新权重
            wt = wt / np.sum(wt)                               # 权重归一化
            self.tree.append(tree)                             # 将当前弱学习器加入列表

    def predict_one(self, x):  # 预测一个数据
        n_estimators = len(self.tree)                          # 获取弱学习器的数量
        fx = 0.0                                               # 存放强学习器预测结果
        for t in range(n_estimators):                          # 遍历各个弱学习器
            y_hat = self.tree[t].predict_one(x)                # 使用各弱学习器进行预测
            fx += self.a[t] * y_hat                            # 计算强学习器预测结果

        return 1 if fx >= 0.0 else -1                          # 返回分类结果

    def predict(self, X): # 预测一个数据集
        y_hat = np.zeros(len(X))                               # 定义预测值向量
        for i in range(len(X)):                                # 遍历每个数据
            y_hat[i] = self.predict_one(X[i])                  # 预测每个数据

        return y_hat                                           # 返回预测结果

    def score(self, X, y):  # 计算预测得分
        y_hat = self.predict(X)                                # 计算数据集X的预测值
        count = np.sum(y_hat == y)                             # 预测值真实值相等的个数

        return count / len(y)                                  # 返回预测得分

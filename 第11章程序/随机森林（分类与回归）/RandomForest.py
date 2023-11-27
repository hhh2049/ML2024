# encoding=utf-8
import numpy as np     # 导入numpy库
from CART import CART  # 导入自实现CART树

class RandomForest:  # 随机森林（分类与回归）的实现
    def __init__(self, X, y, is_classify=True, n_estimators=10, max_depth=None):
        self.X            = X                                     # 训练数据集
        self.y            = y                                     # 训练数据的真实值
        self.is_classify  = is_classify                           # 指明分类还是回归
        self.n_estimators = n_estimators                          # 弱学习器的数量
        self.max_depth    = max_depth                             # 决策树的最大深度
        self.trees        = []                                    # 用于存放弱学习器

    def bootstrap_sample(self):  # bootstrap抽样
        m = len(self.X)                                           # 获取训练样本量
        index = np.random.choice(m, m, replace=True)              # bootstrap抽样

        return np.unique(index)                                   # 返回抽样结果

    def fit(self):  # 拟合数据，训练模型
        for t in range(self.n_estimators):                        # 执行n_estimators次
            sample_index = self.bootstrap_sample()                # bootstrap抽样
            X = self.X[sample_index]                              # 获取抽样数据集
            y = self.y[sample_index]                              # 获取抽样真实值

            tree = CART(X, y, is_classify=self.is_classify,
                        max_depth=self.max_depth)                 # 定义弱学习器对象
            tree.fit()                                            # 训练弱学习器
            self.trees.append(tree)                               # 将弱学习器加入列表

    def predict_one(self, x):  # 预测一个数据
        if self.trees is None: return None                        # 如果树为空则返回

        y_hat = np.zeros(self.n_estimators)                       # 存放弱学习器预测结果
        for t in range(self.n_estimators):                        # 遍历所有弱学习器
            y_hat[t] = self.trees[t].predict_one(x)               # 使用弱学习器进行预测

        if self.is_classify:                                      # 如果用于分类
            labels, counts = np.unique(y_hat, return_counts=True) # 统计各类别及相应数量
            index = np.argmax(counts)                             # 样本数最多类别的下标
            return labels[index]                                  # 返回样本数最多的类别
        else:                                                     # 如果用于回归
            return np.mean(y_hat)                                 # 返回均值

    def predict(self, X):  # 预测一个数据集
        y_hat = np.zeros(len(X))                                  # 用于存放预测结果
        for i in range(len(X)):                                   # 遍历所有数据
            y_hat[i] = self.predict_one(X[i])                     # 预测一个数据

        return y_hat                                              # 返回预测结果

    def score(self, X, y):  # 计算一个数据集的预测得分
        y_hat = self.predict(X)                                   # 预测一个数据集

        if self.is_classify:                                      # 如果用于分类
            count = np.sum(y_hat == y)                            # 预测值真实值相等个数
            return count / len(y)                                 # 计算和返回分类得分
        else:                                                     # 如果用于回归
            diff = y - y_hat                                      # 真实值与预测值之差
            mse = np.dot(diff, diff) / len(X)                     # 计算MSE
            y_mean = np.mean(y)                                   # 真实值的平均值
            diff = y - y_mean                                     # 真实值与平均值之差
            var = np.dot(diff, diff) / len(X)                     # 计算VAR
            return 1.0 - mse / var                                # 计算和返回回归得分

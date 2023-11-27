# encoding=utf-8
import numpy as np
from CART_XGBoost import CART_XGBoost  # 导入自实现的CART树

class XGBoost:  # XGBoost（分类与回归）算法实现
    def __init__(self, X, y,        n_estimators=10,    objective="squarederror",
                 learning_rate=0.1, reg_lambda=1e-5,    gamma=0.0,
                 max_depth=3,       min_sample_split=2, min_child_weight=1.0):
        self.X                = X                         # 训练数据集
        self.y                = y                         # 真实值（离散值或实数值）
        self.n_estimators     = n_estimators              # 弱学习器数量
        self.objective        = objective                 # 目标函数（损失函数）类型
        self.learning_rate    = learning_rate             # 学习率
        self.reg_lambda       = reg_lambda                # 式(12.59)中的正则化参数λ
        self.gamma            = gamma                     # 式(12.59)中的正则化参数γ
        self.max_depth        = max_depth                 # 树的最大深度阈值
        self.min_sample_split = min_sample_split          # 结点最小分裂阈值
        self.min_child_weight = min_child_weight          # 结点最小纯度阈值
        self.tree             = []                        # 用于存储弱学习器

    def fit(self):  # 拟合数据，训练模型
        y_hat = np.zeros(len(self.y))                     # 当前预测值，初始为0
        for t in range(self.n_estimators):                # 依次训练各个弱学习器
            tree = CART_XGBoost(self.X, self.y, y_hat, self.objective,
                                self.reg_lambda, self.gamma, self.max_depth,
                                self.min_sample_split, self.min_child_weight)
            tree.fit()                                    # 训练CART回归树
            self.tree.append(tree)                        # 保存训练好的树
            y_hat_t = tree.predict(self.X)                # 计算当前预测值
            y_hat = y_hat + self.learning_rate * y_hat_t  # 更新总的预测值

    def predict_one(self, x):  # 预测一个数据
        y_hat = 0.0                                       # 用于存放预测值
        for t in range(self.n_estimators):                # 遍历所有弱学习器
            tree = self.tree[t]                           # 获取CART回归树
            y_hat_t = tree.predict_one(x)                 # 执行预测
            y_hat = y_hat + self.learning_rate * y_hat_t  # 计算总的预测值

        return y_hat                                      # 返回预测值

    def predict(self, X):  # 预测一个数据集
        y_hat = np.zeros(len(X))                          # 定义预测值向量
        for i in range(len(X)):                           # 遍历每个数据
            y_hat[i] = self.predict_one(X[i])             # 预测一个数据

        if self.objective == "logistic":                  # 如果是分类问题
            sigmoid = 1.0 / (1.0 + np.exp(-y_hat))        # 计算sigmoid函数值
            y_hat = np.where(sigmoid >= 0.5, 1, 0)        # 根据概率值确定分类

        return y_hat                                      # 返回预测值向量

    def score(self, X, y):  # 计算分类或回归得分
        y_hat = self.predict(X)                           # 获取数据集X的预测值

        if self.objective == "logistic":                  # 如果是分类问题
            count = np.sum(y_hat == y)                    # 预测值与真实值相等的个数
            return count / len(y)                         # 计算并返回分类得分
        elif self.objective == "squarederror":            # 如果是回归问题
            diff = y - y_hat                              # 计算真实值与预测值之差
            mse = np.dot(diff, diff) / len(X)             # 计算MSE
            y_mean = np.mean(y)                           # 计算真实值的平均值
            diff = y - y_mean                             # 计算真实值与平均值之差
            var = np.dot(diff, diff) / len(X)             # 计算VAR
            return 1.0 - mse / var                        # 计算并返回回归得分

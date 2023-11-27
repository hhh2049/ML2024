# encoding=utf-8
import numpy as np
from CART_GBC import CART_GBC

class GradientBoostingClassifier:  # 梯度提升分类树
    def __init__(self, X, y, loss="logistic", n_estimators=10,
                 learning_rate=1.0, max_depth=1):
        self.X             = X                           # 训练数据集
        self.y             = y                           # 训练数据的真实值
        self.loss          = loss                        # 交叉熵损失函数
        self.T             = n_estimators                # 弱学习器的数量
        self.learning_rate = learning_rate               # 学习率
        self.max_depth     = max_depth                   # 树的最大深度
        self.m             = X.shape[0]                  # 训练数据集的数据量
        self.estimators    = []                          # 弱学习器列表

    def fit(self):  # 拟合数据，训练模型
        ft = np.log(np.sum(self.y) / np.sum(1.0-self.y)) # 计算F0
        Ft = np.array([ft] * self.m)                     # 使用F0生成向量
        y_hat = self.y - 1 / (1 + np.exp(-Ft))           # 初始预测值

        for t in range(self.T):                          # 生成各个弱学习器
            tree = CART_GBC(self.X, self.y, y_hat,
                            max_depth=self.max_depth)    # 定义CART_GBC对象
            tree.fit()                                   # 执行训练
            self.estimators.append(tree)                 # 将当前树添加到树列表

            y_hat_t = tree.predict(self.X)               # 使用当前树进行预测
            Ft_1 = Ft + self.learning_rate * y_hat_t     # 计算Ft+1
            Ft = Ft_1                                    # Ft+1作为下轮训练的Ft
            y_hat = self.y - 1 / (1 + np.exp(-Ft))       # 计算残差作为下轮拟合目标

    def predict_one(self, x):  # 预测一个数据
        fx = np.log(np.sum(self.y) / np.sum(1.0-self.y)) # 初始化集成学习器的预测值
        for t in range(self.T):                          # 遍历所有弱学习器
            y_hat_t = self.estimators[t].predict_one(x)  # 使用弱学习器预测
            fx += self.learning_rate * y_hat_t           # 累加预测值

        p = 1 / (1 + np.exp(-fx))                        # 转换为概率
        return 1 if p >= 0.5 else 0                      # 返回分类类别

    def predict(self, X):  # 预测一个数据集
        y_hat = np.zeros(len(X))                         # 定义预测值向量
        for i in range(len(X)):                          # 遍历每个数据
            y_hat[i] = self.predict_one(X[i])            # 预测一个数据

        return y_hat                                     # 返回预测结果

    def score(self, X, y):  # 计算分类得分
        y_hat = self.predict(X)                          # 获取数据集X的预测值
        count = np.sum(y_hat == y)                       # 预测值与真实值相等的个数
        score = count / len(y)                           # 计算得分（分类正确率）

        return score                                     # 返回得分（分类正确率）

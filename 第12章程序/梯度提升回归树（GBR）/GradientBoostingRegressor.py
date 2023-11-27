# encoding=utf-8
import numpy as np
from CART import CART

class GradientBoostingRegressor:  # 梯度提升回归树
    def __init__(self, X, y, loss="ls", n_estimators=10, learning_rate=1.0, max_depth=1):
        self.X            = X                            # 训练数据集
        self.y            = y                            # 训练数据的真实值
        self.loss         = loss                         # 损失函数类型（平方误差函数）
        self.n_estimators = n_estimators                 # 弱学习器数量（决策树的数量）
        self.eta          = learning_rate                # 学习率
        self.max_depth    = max_depth                    # 树的最大深度
        self.estimators   = []                           # 弱学习器列表，一般是回归树

    def fit(self):  # 拟合数据，训练模型
        Ft_1 = np.zeros(len(self.X))                     # 初始化Ft-1(x)=0
        # Ft_1 = np.array([np.mean(self.y)]*len(self.X)) # 初始化Ft-1(x)=均值

        for t in range(self.n_estimators):               # 生成各个弱学习器
            rt = self.y - Ft_1                           # 计算残差
            tree = CART(self.X, rt, is_classify=False,
                        max_depth=self.max_depth)        # 定义回归树（拟合残差）
            tree.fit()                                   # 训练弱学习器
            self.estimators.append(tree)                 # 将当前弱学习器加入列表

            Ft = Ft_1 + self.eta * tree.predict(self.X)  # 更新Ft(x)
            Ft_1 = Ft                                    # Ft(x)作为下一轮Ft-1(x)

    def predict_one(self, x):  # 预测一个数据
        fx = 0.0                                         # 存放强学习器的预测结果
        # fx = np.mean(self.y)                           # 如果训练时初始化为均值

        for t in range(self.n_estimators):               # 遍历所有弱学习器
            y_hat_t = self.estimators[t].predict_one(x)  # 使用弱学习器进行预测
            fx += self.eta * y_hat_t                     # 计算强学习器预测结果

        return fx                                        # 返回预测结果

    def predict(self, X):  # 预测一个数据集
        y_hat = np.zeros(len(X))                         # 定义预测值向量
        for i in range(len(X)):                          # 遍历每个数据
            y_hat[i] = self.predict_one(X[i])            # 预测一个数据

        return y_hat                                     # 返回预测结果

    def score(self, X, y):  # 计算回归得分
        y_hat = self.predict(X)                          # 获取数据集X的预测值
        diff = y - y_hat                                 # 真实值与预测值之差
        mse = np.dot(diff, diff) / len(X)                # 计算MSE
        y_mean = np.mean(y)                              # 真实值的平均值
        diff = y - y_mean                                # 真实值与平均值之差
        var = np.dot(diff, diff) / len(X)                # 计算VAR
        score = 1.0 - mse / var                          # 计算回归得分

        return score                                     # 返回回归得分

    def mse(self, X, y):  # 计算均方误差
        y_hat = self.predict(X)                          # 获取数据集X的预测值
        diff = y - y_hat                                 # 真实值与预测值之差
        mse = np.dot(diff, diff) / len(X)                # 计算MSE

        return mse                                       # 返回MSE

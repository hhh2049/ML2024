# encoding=utf-8
import numpy as np

class Ridge:  # 岭回归的实现
    def __init__(self, X, y, lamda=0.1, fit_type=1, eta=0.01, tol=1e-3, max_iter=3000):
        self.X         = np.c_[np.ones(len(X)), X]   # 训练数据集X，维度m×(n+1)
        self.y         = y                           # 训练数据的真实值y，维度(m,)
        self.lamda     = lamda                       # 正则化参数λ
        self.fit_type  = fit_type                    # 使用哪种方法训练模型
        self.eta       = eta                         # 学习率η
        self.tol       = tol                         # 均方误差变化量阈值
        self.max_iter  = max_iter                    # 最大训练次数
        self.m, self.n = X.shape                     # 获取数据集的数据量和特征数
        self.w         = np.zeros(self.n + 1)        # 待学习的权重向量

    def least_square_method(self):  # 使用最小二乘法直接计算权重向量
        dot_X  = np.dot(self.X.T, self.X)            # 求矩阵乘积
        reg_I  = np.eye(self.n + 1) * self.lamda     # 定义一个很小的正则化对角矩阵
        inv_X  = np.linalg.inv(dot_X + reg_I)        # 求逆矩阵
        dot_X  = np.dot(inv_X, self.X.T)             # 求矩阵乘积
        self.w = np.dot(dot_X, self.y)               # 求矩阵与向量之积

    def stochastic_gradient_descent(self):  # 随机梯度下降法（不含b）
        X, w, y  = self.X, self.w, self.y            # 为缩短下一行代码长度
        last_mse = self.compute_mse(X, w, 0, y)      # 计算初始MSE
        MSE_list = [last_mse]                        # 将初始MSE存入MSE列表
        diff_mse = np.inf                            # MSE变化量，初始为无穷大
        current_iter = 0                             # 当前训练次数

        while diff_mse > self.tol or current_iter < self.max_iter:
            i = np.random.randint(self.m)            # 从0~m-1随机选择一个整数

            y_hat = np.dot(self.X[i], self.w)        # 计算预测值
            grad = (y_hat - self.y[i]) * self.X[i]   # 计算初始梯度
            grad = grad + self.lamda * self.w        # 加入正则化项
            self.w = self.w - self.eta * grad        # 更新权重向量

            mse = self.compute_mse(X, self.w, 0, y)  # 计算当前MSE
            diff_mse = np.abs(last_mse - mse)        # 计算MSE变化量
            MSE_list.append(mse)                     # 将MSE加入列表
            last_mse = mse                           # 保存当前MSE
            current_iter += 1                        # 训练次数加1

        return MSE_list

    def fit(self):  # 拟合数据，训练模型
        if self.fit_type == 0:                       # 拟合类型0，使用最小二乘法
            self.least_square_method()               # 调用最小二乘法
        elif self.fit_type == 1:                     # 拟合类型1，使用随机梯度下降法
            self.stochastic_gradient_descent()       # 调用随机梯度下降法

    def predict(self, X):  # 使用训练好的模型，计算预测值
        X = np.c_[np.ones(len(X)), X]                # 在数据矩阵加入1列1
        y_predict = np.dot(X, self.w)                # 计算预测值

        return y_predict                             # 返回预测值

    def score(self, X, y):  # 使用训练好的模型，计算得分
        y_predict = self.predict(X)                  # 计算预测值
        diff = y - y_predict                         # 计算真实值与预测值之差
        mse = np.dot(diff, diff) / len(X)            # 计算MSE

        y_mean = np.mean(y)                          # 计算真实值的平均值
        diff = y - y_mean                            # 计算真实值与平均值之差
        var = np.dot(diff, diff) / len(X)            # 计算VAR

        return 1.0 - mse / var                       # 返回R平方得分

    @staticmethod
    def compute_mse(X, w, b, y):  # 基于模型参数计算数据集的均方误差
        y_hat = np.dot(X, w) + b                     # 计算预测值向量
        diff  = y_hat - y                            # 计算预测值与真实值之差
        mse   = np.dot(diff, diff) / len(y)          # 计算均方误差

        return mse                                   # 返回均方误差

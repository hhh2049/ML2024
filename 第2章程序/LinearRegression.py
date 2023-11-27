# encoding=utf-8
import numpy as np

class LinearRegression:  # 线性回归的实现
    def __init__(self, X, y, eta=0.1, fit_type=0, k=100, tol=1e-4, max_iter=2000):
        self.X         = X                              # 训练数据集X(m×n)，m个数据n个特征
        self.y         = y                              # 训练数据的真实值y(m,)，m个真实值
        self.eta       = eta                            # 学习率η
        self.fit_type  = fit_type                       # 使用哪种方法训练模型
        self.k         = k                              # 小批量梯度下降的一次训练样本数
        self.tol       = tol                            # 均方误差阈值
        self.max_iter  = max_iter                       # 最大训练次数
        self.m, self.n = X.shape                        # 获取训练数据集的数据量和特征数
        self.w         = np.zeros(self.n)               # 待学习的权重向量(n,)
        self.b         = 0.0                            # 待学习的偏置（标量，一个实数）

    def compute_mse(self):  # 代数形式计算均方误差（MSE）
        result = 0.0                                    # 定义误差平方和的初值

        for i in range(self.m):                         # 对数据集中的每个数据执行如下操作
            y_hat = np.dot(self.X[i], self.w) + self.b  # 计算单个数据的预测值
            square_error = (y_hat - self.y[i]) ** 2     # 计算单个数据的误差平方
            result += square_error                      # 累加到总的误差平方

        return result / self.m                          # 计算并返回均方误差

    @staticmethod  # 静态函数
    def compute_mse2(X, w, b, y):  # 矩阵向量形式计算均方误差（MSE）
        diff = np.dot(X, w) + b - y                     # 计算数据集的误差向量
        return np.dot(diff, diff) / len(y)              # 计算并返回均方误差

    def transform_data(self):  # 数据集X增加1列全为1的向量，用于合并参数w和b
        one_column = np.ones(self.m)                    # 创建一个所有元素都为1的列向量
        X = np.c_[one_column, self.X]                   # 数据集X(m×n)增加1列
        w = np.append(self.b, self.w)                   # 合并参数w和b，w为(n+1)×1列向量

        return X, w                                     # 返回处理后的数据集X和权重向量w

    def least_square_method(self):  # 使用最小二乘法式（2.26）计算权重向量
        X, _  = self.transform_data()                   # 预处理数据：训练数据集X增加1列

        dot_X = np.dot(X.T, X)                          # 求矩阵乘积
        reg   = np.eye(self.n + 1) * 1e-8               # 定义一个很小的正的对角矩阵
        inv_X = np.linalg.inv(dot_X + reg)              # 求逆矩阵
        dot_X = np.dot(inv_X, X.T)                      # 求矩阵乘积
        w     = np.dot(dot_X, self.y)                   # 求权重向量

        self.w = w[1:]                                  # 拆分为w和b，赋值w
        self.b = w[0]                                   # 拆分为w和b，赋值b

    def batch_gradient_descent1(self):  # 批量梯度下降法，基于式（2.36）
        pass  # 作为课后习题

    def batch_gradient_descent2(self):  # 批量梯度下降法，基于式（2.38）
        pass  # 作为课后习题

    def batch_gradient_descent3(self):  # 批量梯度下降法，基于式（2.41）
        pass  # 作为课后习题

    def stochastic_gradient_descent1(self):  # 随机梯度下降法，基于式（2.43）
        last_mse = self.compute_mse()                   # 计算初始均方误差（MSE）
        MSE_list = [last_mse]                           # 将初始MSE存入列表
        diff_mse = np.inf                               # MSE变化量，初始为无穷大
        current_iter = 0                                # 当前训练次数计数
        eta, y, X = self.eta, self.y, self.X            # 为缩短代码长度

        while diff_mse > self.tol or current_iter < self.max_iter:
            i = np.random.randint(self.m)               # 从0到m-1随机选择一个数
            y_hat = np.dot(self.X[i], self.w) + self.b  # 计算预测值

            self.w = self.w - eta * (y_hat-y[i]) * X[i] # 更新w
            self.b = self.b - eta * (y_hat-y[i])        # 更新b

            mse = self.compute_mse()                    # 计算新的均方误差（MSE）
            MSE_list.append(mse)                        # 将新的MSE存入列表
            diff_mse = np.abs(last_mse - mse)           # 计算MSE的变化量（绝对值）
            last_mse = mse                              # 将当前MSE赋值为上一个MSE
            current_iter += 1                           # 训练次数加1

        return MSE_list                                 # 返回均方误差（MSE）列表

    def stochastic_gradient_descent2(self):  # 随机梯度下降法，基于式（2.42）
        X, w = self.transform_data()                    # 训练数据集增加1列，合并参数w和b
        last_mse = self.compute_mse2(X, w, 0, self.y)   # 计算初始MSE
        MSE_list = [last_mse]                           # 将初始MSE存入列表
        diff_mse = np.inf                               # MSE变化量，初始为无穷大
        current_iter = 0                                # 当前训练次数计数
        eta, y = self.eta, self.y                       # 为缩短代码长度

        while diff_mse > self.tol or current_iter < self.max_iter:
            i = np.random.randint(self.m)               # 从0到m-1随机选择一个数
            y_hat = np.dot(X[i], w)                     # 计算预测值

            w = w - eta * (y_hat - y[i]) * X[i]         # 更新w

            mse = self.compute_mse2(X, w, 0, self.y)    # 计算新的MSE
            MSE_list.append(mse)                        # 将新的MSE存入列表
            diff_mse = np.abs(last_mse - mse)           # 计算MSE的变化量
            last_mse = mse                              # 将当前MSE赋值为上一个MSE
            current_iter += 1                           # 训练次数加1

        self.w = w[1:]                                  # 拆封w，赋值给w
        self.b = w[0]                                   # 拆封w，赋值给b
        return MSE_list                                 # 返回均方误差（MSE）列表

    def mini_batch_gradient_descent1(self):  # 小批量梯度下降法，基于式（2.44）
        pass  # 作为课后习题

    def mini_batch_gradient_descent2(self):  # 小批量梯度下降法，基于式（2.45）
        pass  # 作为课后习题

    def mini_batch_gradient_descent3(self):  # 小批量梯度下降法，基于式（2.46）
        pass  # 作为课后习题

    def fit(self):  # 拟合数据，训练模型
        if self.fit_type == 0:                          # 最小二乘法
            self.least_square_method()
        elif self.fit_type == 1:                        # 随机梯度下降法，基于式（2.43）
            self.stochastic_gradient_descent1()
        elif self.fit_type == 2:                        # 随机梯度下降法，基于式（2.42）
            self.stochastic_gradient_descent2()

    def predict(self, X):  # 利用训练好的模型，计算预测值
        y_predict = np.dot(X, self.w) + self.b          # 计算预测值
        return y_predict                                # 返回预测值

    def score(self, X, y):  # 利用训练好的模型，计算评分R方值
        y_predict = np.dot(X, self.w) + self.b          # 计算预测值
        diff = y - y_predict                            # 计算真实值与预测值之差
        mse = np.dot(diff, diff) / len(X)               # 计算MSE

        y_mean = np.mean(y)                             # 计算真实值的平均值
        diff = y - y_mean                               # 计算真实值与平均值之差
        var = np.dot(diff, diff) / len(X)               # 计算VAR

        return 1.0 - mse / var                          # 返回R平方得分，假设var不为0

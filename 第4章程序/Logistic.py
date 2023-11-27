# encoding=utf-8
import numpy as np

class LogisticRegression:  # 逻辑回归的实现
    def __init__(self, X, y, eta=0.1, tol=1e-6, max_iter=1000):
        self.X         = np.c_[np.ones(len(X)), X]        # 训练数据集m×(n+1)
        self.y         = y                                # 真实分类标签(m,),值0或1
        self.eta       = eta                              # 学习率
        self.tol       = tol                              # 训练终止的阈值
        self.max_iter  = max_iter                         # 最大的训练次数
        self.m, self.n = X.shape                          # 数据集的数据量和特征数
        self.w         = np.zeros(self.n + 1)             # 待学习的模型参数

    def compute_loss(self, X, y):  # 基于式（4.16），计算损失函数值
        loss = 0.0                                        # 定义损失函数值
        z = np.dot(X, self.w)                             # 计算预测值（结果为一向量）
        h = 1.0 / (1.0 + np.exp(-z))                      # 计算Logistic函数值

        for i in range(len(X)):                           # 遍历每个数据
            part1 = y[i] * np.log(h[i])                   # X[i]似然第一部分
            part2 = (1 - y[i]) * np.log(1 - h[i])         # X[i]似然第二部分
            xi_loss = part1 + part2                       # 计算当前数据的似然值
            loss += xi_loss                               # 将当前似然值累加到总额

        return -1.0 * loss / len(X)                       # 计算最终的损失函数值

    def stochastic_gradient_descent(self):  # 随机梯度下降法（不含偏置b）
        last_loss = self.compute_loss(self.X, self.y)     # 计算损失函数值
        loss_list = [last_loss]                           # 保存损失函数值
        diff_loss = np.inf                                # 初始损失值变化量为无穷大
        current_iter = 0                                  # 当前训练次数

        X, y, m, eta = self.X, self.y, self.m, self.eta   # 为缩短代码长度
        tol, max_iter = self.tol, self.max_iter           # 为缩短代码长度
        while diff_loss > tol or current_iter < max_iter: # 继续训练的条件
            i = np.random.randint(self.m)                 # 从0~m-1随机选取一个整数

            # 以下为随机梯度下降法，基于式（4.22）
            z = np.dot(X[i], self.w)                      # 计算预测值
            y_hat = 1.0 / (1.0 + np.exp(-z))              # 计算sigmoid函数值
            self.w = self.w - eta * (y_hat - y[i]) * X[i] # 更新权重向量

            # 以下为批量梯度下降法，基于式（4.21）
            # z = np.dot(X, self.w)                       # 计算预测值向量
            # y_hat = 1.0 / (1.0 + np.exp(-z))            # 计算sigmoid函数值向量
            # self.w -= eta * np.dot(X.T, y_hat - y) / m  # 更新权重向量

            loss = self.compute_loss(self.X, self.y)      # 计算当前损失函数值
            diff_loss = np.abs(last_loss - loss)          # 计算损失函数值的变化量
            last_loss = loss                              # 将当前损失值设为上个损失值
            loss_list.append(last_loss)                   # 将当前损失值加入损失值列表
            current_iter += 1                             # 训练次数加1

        return loss_list                                  # 返回损失函数值列表

    def fit(self):  # 拟合数据，训练模型
        self.stochastic_gradient_descent()                # 调用随机梯度下降法训练模型

    def predict(self, X):  # 使用训练好的模型计算预测值
        m = len(X)                                        # 获取数据长度
        X = np.c_[np.ones(m), X]                          # 预处理数据集
        z = np.dot(X, self.w)                             # 计算z值向量
        h = 1.0 / (1 + np.exp(-z))                        # 计算h值向量

        y_hat = np.zeros(m)                               # 定义分类值向量(m,)
        for i in range(m):                                # 遍历X的每个数据
            y_hat[i] = 1 if h[i] >= 0.5 else 0            # 计算分类值，0或1

        return y_hat                                      # 返回数据所属类别向量(m,)

    def score(self, X, y):  # 计算训练或预测得分
        y_hat = self.predict(X)                           # 获取数据集X的预测值
        count = np.sum(y_hat == y)                        # 计数预测值与真实值相等的个数

        return count / len(y)                             # 返回训练或预测得分

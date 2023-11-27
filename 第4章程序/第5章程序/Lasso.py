# encoding=utf-8
import numpy as np

class LassoRegression:  # Lasso回归的实现
    def __init__(self, X, y, fit_type=1, lamda=0.1, tol=1e-3, max_iter=1000):
        self.X         = np.c_[np.ones(len(X)), X]          # 训练数据集X，维度m×(n+1)
        self.y         = y                                  # 训练数据的真实值y，维度(m,)
        self.fit_type  = fit_type                           # 训练哪个模型，LR or Lasso
        self.lamda     = lamda * X.shape[0]                 # 在推导时，用λ除以数据量m
        self.tol       = tol                                # 模型参数w变化量阈值
        self.max_iter  = max_iter                           # 最大训练次数
        self.m, self.n = X.shape                            # 获取数据集的数据量和特征数
        self.w         = np.zeros(self.n + 1)               # 待学习的权重向量

    def compute_delta_w(self, w1, w2):  # 计算模型参数向量w的变化量
        distance_square = np.dot(w1 - w2, w1 - w2)          # 求w1和w2的距离平方
        return np.sqrt(distance_square)                     # 求w1和w2的距离

    def coordinate_descent_for_linear_regression(self):  # 坐标下降法求解线性回归
        last_w = self.w                                     # 保存模型参数向量w
        delta_w = np.inf                                    # 向量w的变化量初值设为无穷大
        current_iter = 0                                    # 当前训练次数计数

        tol, max_iter = self.tol, self.max_iter             # 为缩短代码长度
        while delta_w > tol or current_iter < max_iter:     # 继续训练的条件
            j = current_iter % (self.n + 1)                 # j在0,1,2,...,n之间循环

            pj, zj = 0.0, 0.0                               # pj、zj赋初值0
            w = self.w                                      # 获取当前w的值
            w[j] = 0                                        # 将w[j]设置为0，为计算pj
            X, y = self.X, self.y                           # 为缩短下下一行代码长度
            for i in range(self.m):                         # 遍历所有数据
                pj += (np.dot(X[i], w) - y[i]) * X[i][j]    # 计算pj的值
                zj += self.X[i][j] ** 2                     # 计算zj的值
            wj = -pj / zj                                   # 计算新的wj的值
            self.w[j] = wj                                  # 更新w[j]的值

            delta_w = self.compute_delta_w(self.w, last_w)  # 计算w的变化量
            last_w = self.w                                 # 保存当前的w
            current_iter += 1                               # 训练次数加1

    def coordinate_descent_for_lasso_regression(self):  # 坐标下降法求解Lasso回归
        last_w = self.w                                     # 保存模型参数向量w
        delta_w = np.inf                                    # 向量w的变化量初始设为无穷大
        current_iter = 0                                    # 当前训练次数计数

        tol, max_iter = self.tol, self.max_iter             # 为缩短代码长度
        while delta_w > tol or current_iter < max_iter:     # 继续训练的条件
            j = current_iter % (self.n + 1)                 # j在0,1,...,n之间循环

            pj, zj = 0.0, 0.0                               # pj、zj赋初值0
            w = self.w                                      # 获取当前w的值
            w[j] = 0                                        # 为计算pj
            X, y = self.X, self.y                           # 为缩短代码长度
            for i in range(self.m):                         # 遍历所有数据
                pj += (np.dot(X[i], w) - y[i]) * X[i][j]    # 计算pj的值
                zj += self.X[i][j] ** 2                     # 计算zj的值

            if pj > self.lamda:                             # 根据式（5.28）计算wj值
                wj = (self.lamda - pj) / zj
            elif pj < -1.0 * self.lamda:
                wj = -(self.lamda + pj) / zj
            else:
                wj = 0
            self.w[j] = wj                                  # 更新w[j]的值

            delta_w = self.compute_delta_w(self.w, last_w)  # 计算w的变化量
            last_w = self.w                                 # 保存当前的w
            current_iter += 1                               # 训练次数加1

    def fit(self):  # 拟合数据，训练模型
        if self.fit_type == 0:                              # 拟合类似0，使用坐标下降法
            self.coordinate_descent_for_linear_regression() # 求解线性回归
        elif self.fit_type == 1:                            # 拟合类似1，使用坐标下降法
            self.coordinate_descent_for_lasso_regression()  # 求解Lasso回归

    def predict(self, X):  # 使用训练好的模型，计算预测值
        X = np.c_[np.ones(len(X)), X]                       # 在数据矩阵加入1列1
        y_predict = np.dot(X, self.w)                       # 计算预测值

        return y_predict                                    # 返回预测值

    def score(self, X, y):  # 使用训练好的模型，计算得分
        y_predict = self.predict(X)                         # 计算预测值
        diff = y - y_predict                                # 计算真实值与预测值之差
        mse = np.dot(diff, diff) / len(X)                   # 计算MSE

        y_mean = np.mean(y)                                 # 计算真实值的平均值
        diff = y - y_mean                                   # 计算真实值与平均值之差
        var = np.dot(diff, diff) / len(X)                   # 计算VAR

        return 1.0 - mse / var                              # 返回R平方得分

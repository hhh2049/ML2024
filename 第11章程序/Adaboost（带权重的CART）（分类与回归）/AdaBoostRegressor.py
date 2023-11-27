# encoding=utf-8
import numpy as np
from CART_Weight import CART

class AdaBoostRegressor:  # Adaboost回归的实现
    def __init__(self, X, y, n_estimators=25, loss="linear"):
        self.X            = X                               # 训练数据集
        self.y            = y                               # 训练数据的真实值
        self.n_estimators = n_estimators                    # 弱学习器的数量
        self.loss         = loss                            # 样本误差的计算方式
        self.e            = np.zeros(n_estimators)          # 各弱学习器的错误率
        self.w            = np.zeros(n_estimators)          # 各弱学习器的权重
        self.tree         = []                              # 存放各个弱学习器

    def fit(self):  # 拟合数据，训练模型
        m = self.X.shape[0]                                 # 训练数据集的样本量
        wt = np.array([1/m] * m)                            # 初始化各样本的权重

        for t in range(self.n_estimators):                  # 尝试生成各个弱学习器
            tree = CART(self.X, self.y, is_classify=False,
                        sample_weight=wt, max_depth=5)      # 使用自实现CART回归树
            tree.fit()                                      # 拟合数据，训练模型
            self.tree.append(tree)                          # 将弱学习器加入列表
            y_hat_t = tree.predict(self.X)                  # 使用当前树进行预测

            abs_diff = np.abs(self.y - y_hat_t)             # 所有数据误差的绝对值
            Et = np.max(abs_diff)                           # 获取最大的误差绝对值
            if Et == 0:                                     # 如果误差最大值为0
                self.e[t], self.w[t] = 0.0, 1.0             # 错误率为0，权重为1
                print("Et is 0")                            # 打印提示信息
                break                                       # 完美拟合数据，终止训练
            eti = abs_diff / Et                             # 计算每个数据的错误率
            if self.loss == "square":                       # 如果使用平方误差
                eti = eti ** 2                              # 计算平方误差
            elif self.loss == "exponential":                # 如果使用指数误差
                eti = 1.0 - np.exp(-eti)                    # 计算指数误差
            self.e[t] = np.dot(wt, eti)                     # 计算当前树的错误率
            self.w[t] = self.e[t] / (1 - self.e[t])         # 计算当前学习器权重

            for i in range(m):                              # 遍历所有数据
                wt[i] *= np.power(self.w[t], (1 - eti[i]))  # 计算各个数据新的权重
            wt = wt / np.sum(wt)                            # 新权重归一化

    def predict_one(self, x):  # 预测一个数据
        n_estimators = len(self.tree)                       # 获取弱学习器数量
        if n_estimators == 0: return None                   # 弱学习器数量为0则返回
        self.w = self.w[:n_estimators]                      # 获取有效的权重

        y_hat = np.zeros(n_estimators)                      # 存放各弱学习器预测结果
        for i in range(n_estimators):                       # 遍历各个弱学习器
            y_hat[i] = self.tree[i].predict_one(x)          # 使用弱学习器进行预测

        sorted_index = np.argsort(y_hat)                    # 获取排序后的预测值下标
        smallest_index = 0                                  # 最小的满足不等式的下标
        sum_w = 0.0                                         # 用于存放弱学习器权重和
        threshold = 0.5 * np.sum(self.w)                    # 权重和阈值
        for i in range(n_estimators):                       # 遍历各个弱学习器
            smallest_index = sorted_index[i]                # 获取当前权重下标
            sum_w += self.w[smallest_index]                 # 计算权重之和
            if sum_w >= threshold:                          # 如果权重和超过阈值
                break                                       # 找到目标下标，中止循环
        return y_hat[smallest_index]                        # 返回预测结果

    def predict(self, X):  # 预测一个数据集
        y_hat = np.zeros(len(X))                            # 定义预测值向量
        for i in range(len(X)):                             # 遍历每个数据
            y_hat[i] = self.predict_one(X[i])               # 预测每个数据

        return y_hat                                        # 返回预测结果

    def score(self, X, y):  # 计算训练或预测得分
        y_hat = self.predict(X)                             # 获取数据集X的预测值

        diff = y - y_hat                                    # 真实值与预测值之差
        mse = np.dot(diff, diff) / len(X)                   # 计算MSE
        y_mean = np.mean(y)                                 # 真实值的平均值
        diff = y - y_mean                                   # 真实值与平均值之差
        var = np.dot(diff, diff) / len(X)                   # 计算VAR
        score = 1.0 - mse / var                             # 计算回归得分

        return score                                        # 返回回归得分

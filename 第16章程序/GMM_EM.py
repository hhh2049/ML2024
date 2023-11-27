# encoding=utf-8
import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixture:  # 使用EM算法求解高斯混合模型
    def __init__(self, X, y, n_components, tol=0.0005, reg_covar=1e-6, max_iter=100):
        self.X         = X                                   # 训练数据
        self.y         = y                                   # 训练数据的标签（可为空）
        self.K         = n_components                        # 高斯分布的数量
        self.tol       = tol                                 # 一个非负极小值，收敛阈值
        self.max_iter  = max_iter                            # 最多的训练次数
        self.m, self.n = X.shape                             # 样本量和特征数
        self.reg_covar = np.eye(self.n) * reg_covar          # 正则化协方差，防止奇异矩阵

        self.q         = np.zeros((self.m, self.K))          # E步的q矩阵
        self.w         = np.zeros((self.K, ))                # M步的权重（一维向量）
        self.Mu        = np.zeros((self.K, self.n))          # M步的均值（二维矩阵）
        self.Sigma     = np.zeros((self.K, self.n, self.n))  # M步的协方差（三维张量）

    def initialize_gauss_model(self):  # 初始化各个高斯分布的权重和参数
        for i in range(self.K):                              # 遍历各个高斯分布
            self.w[i]     = 1.0 / self.K                     # 初始时各个高斯分量权重相等
            self.Mu[i]    = self.X[i]                        # 均值向量初始化为前几个数据
            self.Sigma[i] = np.eye(self.n)                   # 协方差矩阵初始化为单位矩阵

    def get_gauss_pdf(self, xi, j):  # 计算某个数据的高斯分布概率密度函数值
        cov = self.Sigma[j] + self.reg_covar                 # 增加正则化项，防止奇异矩阵
        muj = self.Mu[j]                                     # 为缩短下一行代码长度
        pdf = multivariate_normal(mean=muj, cov=cov).pdf(xi) # 计算概率密度函数值

        return pdf                                           # 返回概率密度函数值

    def execute_E(self):  # 执行EM算法的E步
        for i in range(self.m):                              # 遍历所有数据
            N, xi = self.get_gauss_pdf, self.X[i]            # 为缩短代码长度
            p_xi  = 0.0                                      # 用于存放p(xi)
            for j in range(self.K):                          # 遍历各个高斯分布，式(16.57)
                p_xi += self.w[j] * N(xi, j)                 # 计算xi在各类中出现的频率和

            for j in range(self.K):                          # 遍历各个高斯分布
                if p_xi == 0.0:                              # 如果p(xi)为0
                    print("sum_p_xi is zero with i=%d" % i)  # 打印提示信息
                    self.q[i, j] = 1 / self.K                # 用等概率替换
                else:                                        # 如果p(xi)不为0，式(16.59)
                    self.q[i, j] = self.w[j] * N(xi, j)/p_xi # 计算并更新隐变量的后验概率

    def execute_M(self):  # 执行EM算法的M步
        for j in range(self.K):                              # 遍历各个高斯分布
            sum_qij = 0.0                                    # 存放qij之和，i=1,2,...,m
            for i in range(self.m):                          # 遍历所有数据
                sum_qij += self.q[i, j]                      # 计算sum_qij
            if sum_qij == 0:                                 # 如果sum_qij为0
                print("sum_qij_xi is zero with j=%d" % j)    # 打印提示信息
                break                                        # 终止循环

            self.w[j] = sum_qij / self.m                     # 计算并更新各个高斯分布的权重

            sum_qij_xi = np.zeros((self.n, ))                # 存放式（16.67）均值的分子
            for i in range(self.m):                          # 遍历所有数据
                sum_qij_xi += self.q[i, j] * self.X[i]       # 计算式（16.67）均值的分子
            self.Mu[j] = sum_qij_xi / sum_qij                # 计算并更新各个高斯分布的均值

            sum_sigma = np.zeros((self.n, self.n))           # 存放式（16.69）协方差的分子
            for i in range(self.m):                          # 遍历所有数据
                xi, muj = self.X[i], self.Mu[j]              # 为缩短下一行代码的长度
                product = np.outer(xi - muj, xi - muj)       # 计算矩阵乘积
                sum_sigma += self.q[i, j] * product          # 计算协（16.69）方差式的分子
            self.Sigma[j] = sum_sigma / sum_qij              # 计算并更新各高斯分布的协方差

    def get_log_likelihood_value(self):  # 计算对数似然函数值（均值）
        mle = 0.0                                            # 存放对数似然函数值
        for i in range(self.m):                              # 遍历所有数据
            N, xi = self.get_gauss_pdf, self.X[i]            # 为缩短代码长度
            p_xi = 0.0                                       # 存放p(xi)
            for j in range(self.K):                          # 遍历各个高斯分布
                p_xi += self.w[j] * N(xi, j)                 # 计算p(xi)
            mle += np.log(p_xi)                              # 计算对数似然函数值

        return mle / self.m                                  # 返回对数似然函数值（均值）

    def train(self):  # 执行训练
        self.initialize_gauss_model()                        # 初始化模型参数
        last_value = -np.inf                                 # 对数似然函数初始值为负无穷大
        count      = 1                                       # 训练次数计数

        while True:  # 两种方式终止训练：训练达到一定次数或对数似然函数值的增量小于阈值
            self.execute_E()  # 执行E步
            self.execute_M()  # 执行M步

            count += 1                                       # 训练次数计数
            value = self.get_log_likelihood_value()          # 计算对数似然函数值
            if count % 10 == 0: print("count =", count)      # 每训练10次，打印提示信息
            if count == self.max_iter:                       # 达到最大训练次数
                print("count =", count)                      # 打印训练次数
                print("diff  =", value - last_value)         # 打印对数似然函数增量
                break                                        # 终止训练
            if value - last_value < self.tol:                # 对数似然函数增量小于阈值
                print("count =", count)                      # 打印训练次数
                print("diff  =", value - last_value)         # 打印对数似然函数增量
                break                                        # 终止训练
            last_value = value                               # 更新对数似然函数值

    def predict(self, X):  # 判断X中各个数据属于哪个高斯分布（分类类别）
        m, _      = X.shape                                  # 获取数据量
        Q_predict = np.zeros((m, self.K))                    # 每个数据属于每个分布的概率
        N         = self.get_gauss_pdf                       # 为缩短代码长度

        for i in range(m):                                   # 遍历每个数据
            for j in range(self.K):                          # 遍历各个高斯分布
                Q_predict[i][j] = self.w[j] * N(X[i], j)     # 计算加权概率

        y_predict = np.argmax(Q_predict, axis=1)             # 获取每行概率值最大的下标
        return y_predict                                     # 返回预测值

    def print_values(self):  # 打印模型相关参数
        print("\nOur model parameter:")                      # 打印提示信息
        print("w: " + str(self.w))                           # 打印各个高斯分布的权重
        print("Mu: \n" + str(self.Mu))                       # 打印各个高斯分布的均值
        # print("Sigma:\n" + str(self.Sigma))                # 打印各个高斯分布的协方差

# encoding=utf-8
import numpy as np
import scipy.linalg
from sklearn.cluster import KMeans

class SpectralCluster:  # 谱聚类算法实现
    def __init__(self, X, n_clusters=3, affinity="rbf", gamma=1.0):
        self.X          = X                                  # 待聚类的训练数据集
        self.n_clusters = n_clusters                         # 指定的拟聚类的簇数
        self.affinity   = affinity                           # 相似矩阵的构建方式
        self.gamma      = gamma                              # 高斯核函数的参数

        self.m, self.n  = X.shape                            # 获取样本量和维度数
        self.W          = np.zeros((self.m, self.m))         # 邻接矩阵（相似矩阵）
        self.D          = np.zeros((self.m, self.m))         # 度矩阵
        self.L          = np.zeros((self.m, self.m))         # 拉普拉斯矩阵
        self.U          = np.zeros((self.m, n_clusters))     # 特征向量矩阵
        self.labels_    = np.zeros((self.m, ))               # 存储聚类标签

    def fit(self):  # 执行训练
        for i in range(self.m):                              # 遍历所有样本
            for j in range(i, self.m):                       # 从i开始遍历所有样本
                delta = self.X[i] - self.X[j]                # 计算两个样本之差
                distance = np.dot(delta, delta)              # 计算两个样本距离平方
                rbf = np.exp(-1.0 * self.gamma * distance)   # 计算高斯核函数值
                self.W[i][j] = self.W[j][i] = rbf            # 存储边的权重（对称矩阵）

        for i in range(self.m):                              # 遍历所有样本
            self.D[i][i] = np.sum(self.W[i])                 # 生成度矩阵D

        self.L = self.D - self.W                             # 生成拉普拉斯矩阵L
        D_     = scipy.linalg.inv(self.D)                    # 求度矩阵D的逆矩阵
        self.L = np.dot(D_, self.L)                          # 拉普拉斯矩阵L正则化

        values, vectors = scipy.linalg.eig(self.L)           # 求L的特征值和特征向量
        index_sorted = np.argsort(values)                    # 对特征值从小到大排序
        for i in range(self.n_clusters):                     # 对原始数据集进行降维
            vector_i = vectors[:, index_sorted[i]]           # 获取第i个特征向量
            vector_i = np.real(vector_i)                     # 舍去特征向量中的虚数
            self.U[:, i] = vector_i                          # 生成矩阵U的第i列

        model = KMeans(n_clusters=self.n_clusters)           # 定义K均值类对象
        model.fit(self.U)                                    # 对矩阵U进行K均值聚类
        self.labels_ = model.labels_                         # 获取各个样本的簇标记

# encoding=utf-8
import numpy as np
import scipy.linalg

class LaplacianEigenMaps:  # 拉普拉斯特征映射（谱嵌入）算法实现
    def __init__(self, X, n_components=2, affinity="rbf", gamma=1.0):
        self.X            = X                                  # 待降维的数据集
        self.n_components = n_components                       # 降维后的维度数
        self.affinity     = affinity                           # 相似矩阵的构建方式
        self.gamma        = gamma                              # 高斯核函数参数
        self.m, self.n    = X.shape                            # 样本量和特征数
        self.W            = np.zeros((self.m, self.m))         # 邻接矩阵（相似矩阵）
        self.D            = np.zeros((self.m, self.m))         # 度矩阵
        self.L            = np.zeros((self.m, self.m))         # 拉普拉斯矩阵
        self.Y            = np.zeros((self.m, n_components))   # 降维后的数据集

    def fit_transform(self):  # 执行降维操作
        for i in range(self.m):                                # 遍历所有数据
            for j in range(i, self.m):                         # 从i开始遍历所有数据
                delta = self.X[i] - self.X[j]                  # 计算两个数据之差
                distance = np.dot(delta, delta)                # 计算两个数据平方距离
                rbf = np.exp(-1.0 * self.gamma * distance)     # 计算高斯核函数值
                self.W[i][j] = self.W[j][i] = rbf              # 存储边的权重

        for i in range(self.m):                                # 遍历所有数据
            self.D[i][i] = np.sum(self.W[i])                   # 构建度矩阵D

        self.L = self.D - self.W                               # 构建拉普拉斯矩阵L
        D_     = scipy.linalg.inv(self.D)                      # 求度矩阵D的逆矩阵
        self.L = np.dot(D_, self.L)                            # 拉普拉斯矩阵L正则化

        values, vectors = scipy.linalg.eig(self.L)             # 求L的特征值和特征向量
        index_sorted = np.argsort(values)                      # 对特征值从小到大排序
        for j in range(self.n_components):                     # 对原始数据集进行降维
            column_j = np.real(vectors[:, index_sorted[j+1]])  # 舍去特征向量中的虚数
            self.Y[:, j] = column_j                            # 生成矩阵U的第j列

        return self.Y                                          # 返回降维后的数据集

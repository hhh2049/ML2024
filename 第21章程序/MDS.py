# encoding=utf-8
import numpy as np
import scipy.linalg

class MDS:  # 多维标度算法实现
    def __init__(self, X, n_components=2):
        self.X            = X                                # 待降维的数据集
        self.n_components = n_components                     # 降维后的维度数
        self.m, self.n    = X.shape                          # 数据量和特征数
        self.D            = np.zeros((self.m, self.m))       # 距离矩阵D
        self.B            = np.zeros((self.m, self.m))       # 内积矩阵B
        self.X_           = np.zeros((self.m, n_components)) # 降维后的数据集

    def build_matrix_D(self):  # 构建距离矩阵D
        for i in range(self.m):                              # 遍历所有数据
            for j in range(i, self.m):                       # D为对称矩阵，只需计算一半
                delta = self.X[i] - self.X[j]                # 计算两个数据之差
                dist2 = np.dot(delta, delta.T)               # 计算两个数据点的距离平方
                self.D[j][i] = self.D[i][j] = dist2          # 保存两个数据点的距离平方

    def build_matrix_B(self):  # 构建内积矩阵B(仅知距离矩阵D)
        d_i_dot = self.D.sum(axis=1) / self.m                # 矩阵D各行之和
        d_j_dot = self.D.sum(axis=0) / self.m                # 矩阵D各列之和
        d_dot_dot = self.D.sum() / self.m / self.m           # 矩阵D所有元素之和

        for i in range(self.m):                              # 遍历所有数据
            for j in range(self.m):                          # 计算Bij
                temp = self.D[i][j] - d_i_dot[i]             # 计算中间值
                temp = temp - d_j_dot[j] + d_dot_dot         # 计算中间值
                self.B[i][j] = -0.5 * temp                   # 保存Bij

    def build_matrix_B2(self): # 构建内积矩阵B(已知数据集X)
        self.X = self.X - np.mean(self.X, axis=0)            # 数据集X中心化
        for i in range(self.m):                              # 遍历所有数据
            for j in range(i, self.m):                       # B为对称矩阵，只需计算一半
                inner_dot = np.dot(self.X[i], self.X[j])     # 计算两个向量的内积
                self.B[j][i] = self.B[i][j] = inner_dot      # 保存两个向量的内积

    def fit_transform(self):   # 执行降维操作
        self.build_matrix_D()                                # 构建距离矩阵D
        self.build_matrix_B()                                # 构建内积矩阵B
        # self.build_matrix_B2()                             # 构建内积矩阵B(已知数据集X)

        eig_values, eig_vectors = scipy.linalg.eigh(self.B)  # 特征值分解
        V    = np.zeros((self.m, self.n_components))         # 定义矩阵V
        Diag = np.eye(self.n_components)                     # 定义对角矩阵Diag
        index_sorted = np.argsort(-eig_values)               # 特征值从大到小排序

        for i in range(self.n_components):                   # 循环：降维后的维度数
            index = index_sorted[i]                          # 获得降序排序后的下标
            V[:, i] = eig_vectors[:, index].real             # 获取一个特征向量
            Diag[i][i] = np.sqrt(eig_values[index].real)     # 获取一个特征值(平方根)
        self.X_ = np.dot(V, Diag)                            # 生成降维后的数据集

        return self.X_                                       # 返回降维后的数据集(m×k)

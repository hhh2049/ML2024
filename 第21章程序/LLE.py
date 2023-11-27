# encoding=utf-8
import numpy as np
import scipy.linalg

class LocallyLinearEmbedding:  # 局部线性嵌入(LLE)算法实现
    def __init__(self, X, n_neighbors=5, n_components=2, reg=0.001):
        self.X            = X                                  # 待降维的数据集
        self.n_neighbors  = n_neighbors                        # 数据点的近邻数
        self.n_components = n_components                       # 降维后的维度数
        self.reg          = reg                                # 求逆矩阵的正则化项
        self.m, self.n    = X.shape                            # 数据量和特征数
        self.N  = np.zeros((self.m, n_neighbors), dtype="int") # 数据点的近邻下标
        self.W  = np.zeros((self.m, self.m))                   # 数据集的权重矩阵
        self.X_ = np.zeros((self.m, n_components))             # 降维之后的数据集

    def get_neighbors_index_of_data(self):  # 获取每个数据点最近的k个近邻的下标
        D = np.zeros((self.m, self.m))                         # 存放任意两个数据间的距离
        for i in range(self.m):                                # 遍历所有数据
            for j in range(i, self.m):                         # D为对称矩阵，只需计算一半
                delta   = self.X[i] - self.X[j]                # 计算两个数据之差
                D[j][i] = D[i][j] = np.dot(delta, delta.T)     # 计算两个数据间的距离平方

        for i in range(self.m):                                # 获取前k个近邻的下标
            index_sorted = np.argsort(D[i, :])                 # 对距离矩阵的每行排序
            self.N[i, :] = index_sorted[1:self.n_neighbors+1]  # 每行存储数据k个近邻的下标

    def get_neighbors_weight(self, index):  # 计算一个数据点最近的k个近邻的权重
        k = self.n_neighbors                                   # 为缩短代码长度

        X_i = np.zeros((self.n, k))                            # 存储矩阵Xi
        for j in range(k):                                     # 遍历k个近邻
            neighbor_index = int(self.N[index][j])             # 获取一个近邻的下标
            X_i[:, j] = self.X[index] - self.X[neighbor_index] # 计算两个数据点之差

        XiTXi   = np.dot(X_i.T, X_i)                           # 计算矩阵XiTXi
        epsilon = np.eye(k) * self.reg * np.trace(XiTXi)       # 计算正则化项
        XiTXi   = XiTXi + epsilon                              # 防止XiTXi为奇异矩阵
        _XiTXi  = np.linalg.inv(XiTXi)                         # 求XiTXi逆矩阵

        ones_i  = np.ones((k, ))                               # 向量1i
        a       = np.dot(_XiTXi, ones_i)                       # 求权重向量wi的分子
        b       = np.dot(ones_i.T, a)                          # 求权重向量wi的分母
        wi      = a / b                                        # 计算wi

        return wi                                              # 返回wi

    def build_W_matrix(self):  # 计算W矩阵
        self.get_neighbors_index_of_data()                     # 获取每个数据的k个近邻

        for i in range(self.m):                                # 遍历所有数据
            wi = self.get_neighbors_weight(i)                  # 计算wi
            self.W[i][self.N[i]] = wi                          # 将wi赋给矩阵W的第i行

    def fit_transform(self):  # 执行降维操作
        self.build_W_matrix()                                  # 构建W矩阵
        self.W = self.W.T                                      # W的每列为一个求得的wi

        I = np.eye(self.m)                                     # 生成m阶单位矩阵
        M = np.dot(I-self.W, (I-self.W).T)                     # 构建M矩阵
        values, vectors = scipy.linalg.eigh(M)                 # 对M进行特征值分解

        index_sort = np.argsort(values)                        # 对特征值进行排序
        for i in range(self.n_components):                     # 遍历降维后的每一列
            self.X_[:, i] = vectors[:, index_sort[i+1]]        # 对X_每一列进行赋值

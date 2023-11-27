# encoding=utf-8
import numpy as np

class KPCA:  # 核主成分分析算法实现
    def __init__(self, X, n_components=None, kernel="rbf", gamma=1.0):
        self.X            = X                                  # 待降维的数据集
        self.n_components = n_components                       # 降维后的维度数
        self.kernel       = kernel                             # 核函数的类型
        self.gamma        = gamma                              # 核函数的gamma参数值
        self.m, self.n    = X.shape                            # 样本量和维度数
        self.K            = np.zeros((self.m, self.m))         # 核函数矩阵
        self.lambdas_     = None                               # 特征值向量
        self.W            = None                               # 用于降维的矩阵
        self.X_           = None                               # 降维后的数据集

    def get_number_of_dimensions(self):  # 计算降维后的维度数
        if isinstance(self.n_components, int):                 # 若n_components为整型
            if 1 <= self.n_components <= self.n:               # 若维度数在1到n之间
                return                                         # 参数正确，直接返回
            else:                                              # 若维度数不在1到n之间
                self.n_components = self.n                     # 将维度数设置为n
        else:                                                  # 若n_components非整型
            self.n_components = self.n                         # 将维度数设置为n

    def compute_K_matrix(self):  # 计算核函数矩阵
        self.X = self.X - np.mean(self.X, axis=0)              # 数据集中心化
        for i in range(self.m):                                # 遍历所有数据
            for j in range(i, self.m):                         # 从i开始遍历所有数据
                delta = self.X[i] - self.X[j]                  # 计算两个数据之差
                distance = np.dot(delta, delta)                # 计算两个数据平方距离
                rbf = np.exp(-1.0 * self.gamma * distance)     # 计算高斯核函数值
                self.K[i][j] = self.K[j][i] = rbf              # 存储高斯核函数值

        self.K = self.K - np.mean(self.K, axis=0)              # 核函数矩阵中心化

    def fit_transform(self):  # 执行训练（降维）
        self.get_number_of_dimensions()                        # 计算降维后的维度数
        self.compute_K_matrix()                                # 计算核函数矩阵

        eig_values, eig_vectors = np.linalg.eig(self.K)        # 对矩阵K进行特征值分解
        index_k  = np.argsort(-eig_values)[:self.n_components] # 最大的k个特征值下标
        self.lambdas_ = eig_values[index_k]                    # 保存最大的k个特征值
        self.W        = eig_vectors[:, index_k]                # 保存相应的特征向量
        self.X_       = np.zeros((self.m, self.n_components))  # 用于存放降维后的数据

        for i in range(self.m):                                # 遍历所有数据
            for j in range(self.n_components):                 # 遍历前k个维度
                X_ij = np.dot(self.W[:, j], self.K[:, i])      # 计算降维后的Xij
                self.X_[i][j] = X_ij / (eig_values[j] ** 0.5)  # 存储降维后的Xij

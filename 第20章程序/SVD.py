# encoding=utf-8
import numpy as np
from scipy import linalg

class SVD:  # 奇异值分解算法实现
    def __init__(self, X, n_components=None):
        self.X            = X                                  # 待降维的数据集
        self.n_components = n_components                       # 降维后的维度数
        self.m, self.n    = X.shape                            # 样本量和维度数
        self.U            = np.zeros((self.m, self.m))         # 奇异值分解的矩阵U
        self.Sigma        = np.zeros((self.m, self.n))         # 奇异值分解的矩阵Σ
        self.V_T          = np.zeros((self.n, self.n))         # 奇异值分解的矩阵VT
        self.sigma_       = np.zeros(self.n)                   # 存储奇异值向量
        self.sigma_ratio_ = np.zeros(self.n)                   # 奇异值占比向量
        self.X_           = None                               # 降维后的数据集

    def get_number_of_dimensions(self):  # 计算降维后的维度数
        k = self.n_components                                  # 为缩短代码长度
        if not isinstance(k,float) and not isinstance(k,int):  # 若k非浮点数非整数
            self.n_components = self.n                         # 将维度数设置为n

        if isinstance(k, float):                               # 若k为浮点数
            if 0.0 < k < 1.0:                                  # 若k在0到1之间
                sigma_ratio_sum = 0.0                          # 存储奇异值占比之和
                index_all = np.argsort(-self.sigma_ratio_)     # 奇异值占比从大到小排序
                for i in range(self.n):                        # 遍历所有维度
                    j = index_all[i]                           # 第i大的奇异值占比下标
                    sigma_ratio_sum += self.sigma_ratio_[j]    # 计算奇异值占比之和
                    if sigma_ratio_sum >= k:                   # 奇异值占比超过阈值
                        self.n_components = i + 1              # 设置降维后的维度数
                        break                                  # 中止循环
            else:                                              # 若k不在0到1之间
                self.n_components = self.n                     # 将维度数设置为n

        if isinstance(k, int):                                 # 若k为整型
            if 1 <= k <= self.n:                               # 若k在1到n之间
                return                                         # 参数正确，直接返回
            else:                                              # 若k不在1到n之间
                self.n_components = self.n                     # 将维度数设置为n

    def fit_transform(self):  # 执行训练（降维）
        XTX = np.dot(self.X.T, self.X)                         # 计算矩阵XT*X
        eig_values, eig_vectors = np.linalg.eig(XTX)           # 对XTX进行特征值分解
        index_all   = np.argsort(-eig_values)                  # 用于特征值从大到小排序
        eig_values  = eig_values[index_all]                    # 对特征值从大到小排序
        eig_vectors = eig_vectors[:, index_all]                # 相应调整特征向量次序
        self.sigma_ = np.sqrt(eig_values)                      # 计算奇异值向量
        self.sigma_ratio_ = self.sigma_ / np.sum(self.sigma_)  # 计算奇异值占比向量
        self.get_number_of_dimensions()                        # 计算降维后的维度数

        r = 0                                                  # 用于存放矩阵A的秩
        for i in range(self.n):                                # 遍历所有特征
            self.Sigma[i][i] = eig_values[i] ** 0.5            # 计算并保存A的奇异值
            if eig_values[i] > 0: r += 1                       # 计算并更新矩阵A的秩
        self.V_T = eig_vectors.T                               # 计算并保存VT

        for i in range(r):                                     # 遍历r次
            U, Sigma, V_T = self.U, self.Sigma, self.V_T       # 为缩短代码长度
            U[:, i] = 1 / Sigma[i][i] * np.dot(self.X, V_T[i]) # 计算U1并添加到U

        U2 = linalg.null_space(self.X.T)                       # 计算U2
        for i in range(self.m - r):                            # 遍历m-r次
            self.U[:, r + i] = U2[:, i]                        # 将U2添加到U

        V = self.V_T.T                                         # 计算矩阵V
        self.X_ = np.dot(self.X, V[:, :self.n_components])     # 计算降维后的数据集X

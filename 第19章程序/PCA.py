# encoding=utf-8
import numpy as np

class PCA:  # 主成分分析算法实现
    def __init__(self, X, n_components=None):  # n_components为整数或0到1之间的小数
        self.X              = X                                # 待降维的数据集
        self.n_components   = n_components                     # 降维后的维度数
        self.m, self.n      = X.shape                          # 样本量和维度数
        self.W              = None                             # 用于降维的矩阵
        self.lambdas_       = np.zeros(self.n)                 # 存储特征值向量
        self.lambdas_ratio_ = np.zeros(self.n)                 # 特征值占比向量
        self.X_             = None                             # 降维后的数据集

    def get_number_of_dimensions(self):  # 计算降维后的维度数
        k = self.n_components                                  # 为缩短代码长度
        if not isinstance(k,float) and not isinstance(k,int):  # 若k非浮点数非整数
            self.n_components = self.n                         # 将维度数设置为n

        if isinstance(k, float):                               # 若维度数为浮点数
            if 0.0 < k < 1.0:                                  # 若维度数在0到1之间
                lambda_ratio_sum = 0.0                         # 存储特征值占比之和
                for i in range(self.n):                        # 遍历所有维度
                    lambda_ratio_sum += self.lambdas_ratio_[i] # 计算特征值占比之和
                    if lambda_ratio_sum >= k:                  # 特征值占比大于阈值
                        self.n_components = i + 1              # 设置降维后的维度数
                        break                                  # 中止循环
            else:                                              # 若维度数不在0到1之间
                self.n_components = self.n                     # 将维度数设置为n

        if isinstance(k, int):                                 # 若维度数为整型
            if 1 <= k <= self.n:                               # 若维度数在1到n之间
                return                                         # 参数正确，直接返回
            else:                                              # 若维度数不在1到n之间
                self.n_components = self.n                     # 将维度数设置为n

    def fit_transform(self):  # 执行训练（降维）
        self.X   = self.X - np.mean(self.X, axis=0)            # 数据集中心化
        C_matrix = np.dot(self.X.T, self.X) / self.n           # 计算C矩阵

        eig_values, eig_vectors = np.linalg.eig(C_matrix)      # 特征值分解
        self.lambdas_ratio_ = eig_values / np.sum(eig_values)  # 计算各个特征值占比
        self.get_number_of_dimensions()                        # 计算降维后的维度数

        index_all           = np.argsort(-eig_values)          # 特征值从大到小排序
        index_k             = index_all[:self.n_components]    # 取前k个特征值下标
        self.lambdas_       = eig_values[index_k]              # 存储前k个特征值
        self.lambdas_ratio_ = self.lambdas_ratio_[index_k]     # 存储前k个特征值占比
        self.W              = eig_vectors[:, index_k]          # 构造用于降维的矩阵
        self.X_             = np.dot(self.X, self.W)           # 对原始数据集进行降维

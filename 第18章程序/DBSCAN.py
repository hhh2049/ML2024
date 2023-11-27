# encoding=utf-8
import numpy as np
from random import choice

class DBSCAN:  # 基于密度的聚类算法（DBSCAN）实现
    def __init__(self, X, eps=0.5, min_samples=5):
        self.X           = X                                 # 待聚类的训练数据集
        self.epsilon     = eps                               # 邻域的半径阈值ε
        self.min_samples = min_samples                       # 邻域的样本阈值MinPts
        self.m, self.n   = X.shape                           # 获取样本量和维度数
        self.n_cluster   = 0                                 # 聚类形成的簇数
        self.labels_     = np.full((self.m, ), -1)           # 各个样本所属簇的标签

    def initialize(self):  # 计算样本之间的距离，找出所有核心点
        D = np.full((self.m, self.m), 0.0)                   # 任意两个样本之间的距离
        for i in range(self.m):                              # 遍历所有样本
            for j in range(i + 1, self.m):                   # 从i+1开始遍历所有样本
                delta = self.X[i] - self.X[j]                # 计算两个样本之差
                distance = np.dot(delta, delta) ** 0.5       # 计算两个样本之间的距离
                D[i][j] = D[j][i] = distance                 # 保存两个样本之间的距离

        is_core = np.full((self.m, ), False)                 # 用于标记样本是否为核心点
        for i in range(self.m):                              # 遍历所有样本
            count = 0                                        # 邻居点计数
            for j in range(self.m):                          # 遍历所有样本
                if D[i][j] <= self.epsilon: count += 1       # 如果为邻居点，则计数加一
                if count >= self.min_samples: break          # 如果邻居点达到阈值则中止
            if count >= self.min_samples:                    # 如果邻居点达到阈值
                is_core[i] = True                            # 将当前样本标记为核心点

        return D, is_core                                    # 返回距离矩阵和核心点向量

    def is_clustered(self, is_core):  # 判断所有核心点是否均已聚类
        for i in range(self.m):                              # 遍历所有样本
            if is_core[i] and self.labels_[i] == -1:         # 当前样本为核心点且未标记
                return False, i                              # 返回否和一个未聚类核心点

        return True, -1                                      # 返回已完成聚类的标志

    def get_neighbors(self, i, D):  # 获取一个核心点邻域内所有样本
        neighbors = set()                                    # 用于存放邻居点的集合
        for j in range(self.m):                              # 遍历所有样本
            if D[i][j] <= self.epsilon and j != i:           # 如果当前样本为邻居点
                neighbors.add(j)                             # 加入到邻居点集合

        return neighbors                                     # 返回X[i]的所有邻居点

    def fit(self):  # 执行训练
        D, cores = self.initialize()                         # 初始化

        current_cluster = 0                                  # 当前簇标记：0～n_cluster-1
        done, current_core_i = self.is_clustered(cores)      # 判断是否已完成聚类
        while not done:                                      # 有未聚类的核心点则继续聚类
            current_core_set = set()                         # 用于存放当前簇的所有核心点
            current_core_set.add(current_core_i)             # 将当前核心点加入到当前簇中

            while len(current_core_set) > 0:                 # 如果当前簇的核心点集合非空
                i = choice(list(current_core_set))           # 从集合中随机选择一个核心点
                self.labels_[i] = current_cluster            # 对选出的核心点赋予簇标记
                neighbors = self.get_neighbors(i, D)         # 找出该核心点的所有邻居点

                for j in neighbors:                          # 遍历当前核心点的邻居点
                    if cores[j] and self.labels_[j] == -1:   # 如果为核心点且未进行标记
                        current_core_set.add(j)              # 加入到当前簇的核心点集合
                    if not cores[j]:                         # 如果为普通邻居点
                        self.labels_[j] = current_cluster    # 将其标记为当前簇

                current_core_set.remove(i)                   # 从当前簇核心点集合删除该点

            current_cluster += 1                             # 处理完一个簇，簇标记加1
            done, current_core_i = self.is_clustered(cores)  # 判断是否已完成聚类

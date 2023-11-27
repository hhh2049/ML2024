# encoding=utf-8
import numpy as np
import random

class K_Means:  # K均值聚类算法实现
    def __init__(self, X, n_clusters=3, epsilon=1e-4):
        self.X            = X                                  # 待聚类的训练数据
        self.n_clusters   = n_clusters                         # 拟聚类的簇数K
        self.epsilon      = epsilon                            # 各簇质心变化的阈值
        self.m, self.n    = X.shape                            # 数据的样本量和维度数
        self.init_centers = np.zeros((n_clusters, self.n))     # 各簇的质心（初始化时）
        self.centers_     = np.zeros((n_clusters, self.n))     # 各簇的质心（训练时）
        self.labels_      = np.full((self.m, ), -1)            # 各样本所属簇的标签
        self.inertia_     = 0.0                                # 各样本到簇质心的距离和

    def get_nearest_center_index(self, x): # 获取距样本点最近的簇
        min_distance = np.inf                                  # 样本点到各簇的最近距离
        center_index = -1                                      # 样本点所属簇的下标

        for i in range(self.n_clusters):                       # 遍历各个簇
            mu_i     = self.centers_[i]                        # 获取簇质心
            distance = np.dot(x - mu_i, x - mu_i)              # 计算样本点到簇的距离
            if distance < min_distance:                        # 如果当前距离最小
                min_distance = distance                        # 更新最小距离
                center_index = i                               # 更新簇的下标（标签）

        return center_index                                    # 返回样本点所属簇的下标

    def get_new_center_list(self):  # 重新计算各个簇的质心
        center_list = np.zeros((self.n_clusters, self.n))      # 存放各个簇的质心
        counts_list = np.zeros((self.n_clusters, ))            # 存放各个簇的所含样本数

        for i in range(self.m):                                # 遍历所有样本
            index = self.labels_[i]                            # 获取当前样本所属的簇
            center_list[index] += self.X[i]                    # 累加数据
            counts_list[index] += 1                            # 累加计数

        for i in range(self.n_clusters):                       # 遍历所有的簇
            center_list[i] = center_list[i] / counts_list[i]   # 计算各簇质心

        return center_list                                     # 返回各簇质心

    def is_center_changed(self, new_center_list):  # 判断各簇质心是否有变
        changed_amount = 0.0                                   # 存放所有簇质心的变化量
        for i in range(self.n_clusters):                       # 遍历所有的簇
            delta = new_center_list[i] - self.centers_[i]      # 计算新旧簇质心的差值
            changed_amount += np.dot(delta, delta) ** 0.5      # 累加簇质心的变化量

        if changed_amount < self.epsilon:                      # 簇质心的变化量小于阈值
            return False                                       # 返回簇质心无变化
        else:                                                  # 否则
            return True                                        # 返回簇质心有变化

    def get_inertia(self):  # 计算各样本到簇质心的距离和
        for i in range(self.m):                                # 遍历所有样本
            index = self.labels_[i]                            # 获取当前样本所属的簇
            delta = self.X[i] - self.centers_[index]           # 计算样本与簇质心之差
            self.inertia_ += np.dot(delta, delta)              # 累加样本到簇质心距离

    def fit(self): # 执行训练
        index = [i for i in range(self.m)]                     # 获取所有数据下标
        center_index = random.sample(index, self.n_clusters)   # 随机选择K个下标
        for i in range(self.n_clusters):                       # 遍历所有的簇
            self.centers_[i]     = self.X[center_index[i]]     # 初始化簇质心
            self.init_centers[i] = self.X[center_index[i]]     # 保存初始化簇质心

        is_changed = True                                      # 簇质心是否发生变化的标志
        while is_changed:                                      # 簇质心发生变化则继续训练
            for i in range(self.m):                            # 遍历所有的样本数据
                c = self.get_nearest_center_index(self.X[i])   # 判断样本属于哪个簇
                self.labels_[i] = c                            # 将样本标记为相应簇

            new_centers = self.get_new_center_list()           # 重新计算各个簇的质心
            is_changed  = self.is_center_changed(new_centers)  # 判断各簇质心是否改变
            self.centers_ = new_centers                        # 更新各个簇的质心

        self.get_inertia()                                     # 计算各样本到簇中心的距离和

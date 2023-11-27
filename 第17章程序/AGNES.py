# encoding=utf-8
import numpy as np

class AGNES:  # 层次聚类算法(AGNES)实现
    def __init__(self, X, n_clusters=3, linkage="single"):
        self.X          = X                                   # 待聚类的训练数据集
        self.n_clusters = n_clusters                          # 指定的拟聚类的簇数
        self.linkage    = linkage                             # 簇间距离的计算方式
        self.m, self.n  = X.shape                             # 获取样本量和维度数
        self.labels_    = np.full((self.m,), -1)              # 各个样本所属簇的标签
        self.D          = np.zeros((self.m, self.m))          # 任意两个样本间的距离

    def get_sample_distance(self):  # 计算样本之间的距离（平方）
        for i in range(self.m):                               # 遍历所有样本
            for j in range(i+1, self.m):                      # 从i+1开始遍历样本
                delta    = self.X[i] - self.X[j]              # 计算两个样本之差
                distance = np.dot(delta, delta)               # 计算两个样本间的距离
                self.D[i][j] = self.D[j][i] = distance        # 保存两个样本间的距离

    def get_cluster_distance(self, c1_list, c2_list):  # 计算两个簇之间的距离
        min_distance = np.inf                                 # 簇间距离初值为无穷大

        for i in c1_list:                                     # 遍历簇1所有样本
            for j in c2_list:                                 # 遍历簇2所有样本
                distance = self.D[i][j]                       # 获取两个样本间的距离
                if distance < min_distance:                   # 如果当前距离更小
                    min_distance = distance                   # 更新簇间距离

        return min_distance                                   # 返回簇间距离

    def get_two_nearest_cluster(self, cluster_list):  # 获取距离最近的两个簇
        min_distance = np.inf                                 # 距离最近的两个簇的距离
        min_c1_index = min_c2_index = -1                      # 距离最近的两个簇的下标

        for i in range(len(cluster_list)):                    # 遍历所有簇
            for j in range(i+1, len(cluster_list)):           # 从i+1开始遍历所有簇
                c1, c2 = cluster_list[i], cluster_list[j]     # 获取两个簇的所有下标
                distance = self.get_cluster_distance(c1, c2)  # 计算两个簇之间的距离
                if distance < min_distance:                   # 若当前两个簇距离更小
                    min_distance = distance                   # 更新最小距离
                    min_c1_index, min_c2_index = i, j         # 更新簇的下标

        return min_c1_index, min_c2_index                     # 返回最近的两个簇的下标

    def fit(self):  # 执行训练
        self.get_sample_distance()                            # 计算所有样本之间的距离
        cluster_list = []                                     # 用于存放所有簇的列表
        for i in range(self.m):                               # 初始时每个样本为一个簇
            cluster_list.append([i])                          # 簇中存放所含样本的下标

        while len(cluster_list) != self.n_clusters:           # 当前簇的数量未到阈值
            i, j = self.get_two_nearest_cluster(cluster_list) # 获取距离最近的两个簇

            cluster_list[i] += cluster_list[j]                # 合并两个簇所含样本下标
            cluster_list.remove(cluster_list[j])              # 删除被合并的簇

        for i in range(len(cluster_list)):                    # 为每个样本设置簇标签
            for index in cluster_list[i]:                     # 遍历一个簇的所有样本
                self.labels_[index] = i                       # 为当前样本设置簇标签

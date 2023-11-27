# encoding=utf-8
import numpy as np
from tqdm import tqdm

class IsometricMapping:  # 等度量映射算法实现
    def __init__(self, X, n_neighbors=10, n_components=2):
        self.X            = X                                 # 待降维的数据集
        self.n_neighbors  = n_neighbors                       # 数据点的近邻数
        self.n_components = n_components                      # 降维后的维度数
        self.m, self.n    = X.shape                           # 数据量和特征数
        self.D            = np.zeros((self.m, self.m))        # 距离矩阵D
        self.B            = np.zeros((self.m, self.m))        # 内积矩阵B
        self.Z            = np.zeros((self.m, n_components))  # 降维后的数据

    def build_matrix_D(self):  # 构建距离矩阵D
        Graph = np.zeros((self.m, self.m))                    # 用于存储邻接矩阵

        # 计算任意两点间的欧氏距离
        for i in range(self.m):                               # 遍历所有数据
            for j in range(i+1, self.m):                      # 对称矩阵，只需计算一半
                delta = self.X[i] - self.X[j]                 # 计算两个数据之差
                dist = np.dot(delta, delta.T) ** 0.5          # 计算两个数据点的距离
                Graph[j][i] = Graph[i][j] = dist              # 保存两个数据点的距离

        # 每个点到前k个近邻为欧氏距离，到其他点的距离为无穷大
        for i in range(self.m):                               # 遍历所有数据
            index_sorted = np.argsort(Graph[i, :])            # 按距离从小到大进行排序
            index_sorted = index_sorted[:self.n_neighbors+1]  # 选择前k+1个近邻(含自身)
            for j in range(self.m):                           # 遍历所有数据
                if j not in index_sorted:                     # 若xj不是xi的近邻
                    Graph[i][j] = np.inf                      # 距离设置为无穷大

        # 邻接矩阵Graph对称化
        for i in range(self.m):                               # 遍历所有数据
            for j in range(i + 1, self.m):                    # 从i+1开始遍历数据
                if Graph[i][j] != Graph[j][i]:                # 如果矩阵不对称
                    if Graph[i][j] > Graph[j][i]:             # 如果Graph[i][j]值更大
                        Graph[i][j] = Graph[j][i]             # 修改元素的值使之对称
                    else:                                     # 如果Graph[j][i]值更大
                        Graph[j][i] = Graph[i][j]             # 修改元素的值使之对称

        # 调用迪杰斯特拉算法，计算任意两点间的测地线距离
        for i in tqdm(range(self.m)):                         # 使用tqdm()显示进度
            self.D[i, :] = self.dijkstra(i, Graph)            # 计算xi到其他点的测地距离

    def build_matrix_B(self):  # 构建内积矩阵B
        self.D = self.D ** 2                                  # 计算距离的平方

        d_i_dot   = self.D.sum(axis=1) / self.m               # 矩阵D各行之和
        d_j_dot   = self.D.sum(axis=0) / self.m               # 矩阵D各列之和
        d_dot_dot = self.D.sum() / self.m / self.m            # 矩阵D所有元素之和

        for i in range(self.m):                               # 遍历所有数据
            for j in range(self.m):                           # 计算Bij
                temp = self.D[i][j] - d_i_dot[i]              # 计算中间值
                temp = temp - d_j_dot[j] + d_dot_dot          # 计算中间值
                self.B[i][j] = -0.5 * temp                    # 保存Bij

    def fit_transform(self):  # 执行降维操作
        self.build_matrix_D()                                 # 构建距离矩阵D
        self.build_matrix_B()                                 # 构建内积矩阵B

        eig_values, eig_vectors = np.linalg.eig(self.B)       # 对矩阵B进行特征值分解
        index_sorted = np.argsort(-eig_values)                # 特征值从大到小排序
        index_sorted = index_sorted[:self.n_components]       # 截取前几个特征值
        V = eig_vectors[:, index_sorted].real                 # 构建矩阵V
        Diag = np.diag(eig_values[index_sorted].real ** 0.5)  # 构建对角矩阵
        self.Z = np.dot(V, Diag)                              # 生成降维后的数据集

        return self.Z                                         # 返回降维后的数据集

    @staticmethod  # 静态函数
    def dijkstra(start_vertex, graph):  # 计算start_vertex到所有结点的最短距离
        passed = [start_vertex]                               # 已找到最短距离的顶点
        unpass = []                                           # 未找到最短距离的顶点
        for i in range(len(graph)):                           # 遍历所有顶点
            if i != start_vertex:                             # 如果不是起始结点
                unpass.append(i)                              # 将该结点加入到unpass
        dist = graph[start_vertex]                            # 初始到所有点的最短距离

        while len(unpass):                                    # 如果unpass非空则继续
            index = unpass[0]                                 # 选择unpass第一个结点
            for i in unpass:                                  # 遍历unpass中所有结点
                if dist[i] < dist[index]:                     # 若到当前结点距离更短
                    index = i                                 # 选中结点i

            unpass.remove(index)                              # 从unpass中删除index
            passed.append(index)                              # 将index加入到passed

            for i in unpass:                                  # 遍历unpass所有结点
                if dist[index] + graph[index][i] < dist[i]:   # 若经结点index距离更短
                    dist[i] = dist[index] + graph[index][i]   # 更新当前结点到i的距离

        return dist                                           # 返回到各个结点最短距离

# encoding=utf-8
import numpy as np
from collections import Counter

class LDA:  # 线性判别分析算法实现
    def __init__(self, X, y, solver="eigen", n_components=None):
        self.X            = X                               # 训练数据集
        self.y            = y                               # 真实分类标签
        self.solver       = solver                          # 求解器的类型
        self.n_components = n_components                    # 降维后的维数
        self.m, self.n    = X.shape                         # 获取数据量和特征数
        self.c            = len(Counter(y))                 # 获取数据集的类别数
        self.mean_        = np.zeros((self.c, self.n))      # 每个类别的均值向量
        self.mu           = np.zeros((self.n, ))            # 整体样本的均值向量
        self.Sw           = np.zeros((self.n, self.n))      # 类内散布矩阵
        self.Sb           = np.zeros((self.n, self.n))      # 类间散布矩阵
        self.W            = np.zeros((self.n, self.c-1))    # 降维权重矩阵

    def compute_model_values(self):  # 计算类内散布矩阵Sw和计算类间散布矩阵Sb
        self.mu = np.mean(self.X, axis=0)                   # 计算整体样本均值
        counter = Counter(self.y)                           # 获取分类标签列表
        for class_label in counter:                         # 遍历所有类别标签
            X_class_label = self.X[self.y == class_label]   # 根据标签筛选样本
            u_i = np.mean(X_class_label, axis=0)            # 计算本类样本均值
            self.mean_[class_label] = u_i                   # 保存本类样本均值

            for x in X_class_label:                         # 遍历本类所有样本
                diff = (x - u_i).reshape(self.n, 1)         # 计算样本均值之差
                self.Sw = self.Sw + np.dot(diff, diff.T)    # 计算类内散布矩阵

            m_i = len(X_class_label)                        # 获取本类样本数量
            diff = (u_i - self.mu).reshape(self.n, 1)       # 计算类间均值之差
            self.Sb = self.Sb + m_i * np.dot(diff, diff.T)  # 计算类间散布矩阵

    def fit(self):  # 拟合数据训练模型
        self.compute_model_values()                         # 计算模型相关的值

        reg = np.eye(self.n) * 1e-8                         # 待添加正则化矩阵
        _Sw = np.linalg.inv(self.Sw + reg)                  # 类内散布矩阵求逆
        Sp  = np.dot(_Sw, self.Sb)                          # 生成待分解的矩阵
        values, vectors = np.linalg.eig(Sp)                 # 特征值分解

        index_sort = np.argsort(-values)                    # 特征值从大到小排序
        for i in range(self.c-1):                           # 获取前c-1个特征向量
            self.W[:, i] = vectors[:, index_sort[i]]        # 生成权重矩阵

    def predict(self, X):  # 预测样本所属的类别
        means = np.zeros((self.c, self.c-1))                # 存放投影后各类中心点
        for i in range(self.c):                             # 遍历所有类别
            means[i] = np.dot(self.W.T, self.mean_[i])      # 计算投影后各类中心点

        y_hat = np.zeros((len(X), ), dtype="int")           # 存放数据的预测类别
        for i in range(len(X)):                             # 遍历所有数据
            z = np.dot(self.W.T, X[i])                      # 计算投影后的坐标

            min_distance, label = np.inf, -1                # 存放最小距离和所属标签
            for j in range(self.c):                         # 遍历所有类别
                diff = z - means[j]                         # 投影与第j类中心点之差
                distance = np.dot(diff, diff)               # 投影与第j类中心点距离
                if distance < min_distance:                 # 如果当前距离更小
                    min_distance, label = distance, j       # 最小距离和所属标签赋值
            y_hat[i] = label                                # 保存当前数据点所属类别

        return y_hat                                        # 返回数据集的预测类别

    def score(self, X, y):  # 计算预测分类的准确率
        y_hat = self.predict(X)                             # 预测各样本所属的类
        count = np.sum(y_hat == y)                          # 预测值与真实值相等的个数

        return count / len(y)                               # 返回预测正确率

    def transform(self, X):  # 对输入数据进行投影（降维）
        if self.n_components <= self.c - 1:                 # 校验拟降维的维度数
            W = -1.0 * self.W[:, :self.n_components]        # 数乘特征向量仍正确
            Z = np.zeros((len(X), self.c-1))                # 存放降维后的数据集
            for i in range(len(X)):                         # 遍历X的所有数据
                Z[i] = np.dot(W.T, X[i])                    # 计算降维后的数据
            return Z                                        # 返回降维后的数据集
        else:
            print("Projection has wrong dimensions")        # 投影降维的维度有误
            return None                                     # 返回空

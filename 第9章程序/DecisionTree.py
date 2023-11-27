# encoding=utf-8
import numpy as np

class Node:  # 决策树的结点类
    def __init__(self, D=None, n_set=None):
        self.D        = D       # 包含真实值的当前数据集，也可以只存储原始数据集的下标
        self.y_hat    = None    # 该结点的预测值，该结点数量最多的类的标签
        self.n_set    = n_set   # 该结点可用于划分的特征集合
        self.n_split  = None    # 该结点用于划分的特征的下标，若值为None则为叶子结点
        self.children = []      # 该结点的子结点，列表元素为元组(特征的值，子结点)

class DecisionTree:  # 决策树模型（ID3算法）的实现（仅用于分类，特征的值为离散值）
    def __init__(self, X, y, epsilon=0.0, criterion="entropy"):
        self.X         = np.c_[X, y]                           # 分类标签并入数据集
        self.m, self.n = X.shape                               # 获取数据量和特征数
        self.epsilon   = epsilon                               # 信息增益的阈值
        self.criterion = criterion                             # 选择特征的方法
        self.tree      = None                                  # 存储生成的决策树

    def fit(self):  # 拟合数据，训练模型
        n_set = set(i for i in range(self.n))                  # 可用于划分的特征集合
        X, epsilon = self.X, self.epsilon                      # 为缩短下一行代码长度
        self.tree = build_decision_tree(X, n_set, epsilon)     # 构建决策树

    def predict_one(self, x):  # 预测一个数据
        if self.tree is None: return None                      # 若决策树为空则返回

        y_hat = None                                           # 存储预测值
        node = self.tree[0]                                    # 将根结点作为当前结点
        while node is not None:                                # 当前结点非空，继续查找
            y_hat = node.y_hat                                 # 获取当前结点的预测值
            if node.n_split is None: break                     # 若为叶子节点，则终止
            is_find = False                                    # 标记是否找到子结点
            for value, child in node.children:                 # 遍历子结点
                if x[node.n_split] == value:                   # 若为相应子结点
                    node = child                               # 将子结点作为当前结点
                    is_find = True                             # 找到标记置为真
                    break                                      # 终止内层循环
            if not is_find: break                              # 未找到相应子结点则终止

        return y_hat                                           # 返回预测值

    def predict(self, X):  # 预测一个数据集
        y_hat = np.zeros(len(X))                               # 定义存储预测值的向量
        for i in range(len(X)):                                # 遍历每个数据
            y_hat[i] = self.predict_one(X[i])                  # 预测一个数据

        return y_hat                                           # 返回预测结果

    def score(self, X, y):  # 计算分类得分
        y_hat = self.predict(X)                                # 计算预测值
        count = np.sum(y_hat == y)                             # 预测值与真实值相等的个数

        return count / len(y)                                  # 计算分类得分

def build_decision_tree(D, n_set, e):  # 构建分类决策树（特征为离散值）
    tree = []                                                  # 用列表存储生成的树
    root = Node(D, n_set)                                      # 定义当前树的根结点
    root.y_hat = get_max_label(D)                              # 当前结点的分类标签
    tree.append(root)                                          # 将根结点加入到树中

    if len(np.unique(D[:, -1])) == 1: return tree              # 若数据为同一类返回
    if len(n_set) == 0: return tree                            # 若可用特征为空返回

    j, gain = select_split_feature(D, n_set)                   # 选出用于划分的特征
    if gain < e: return tree                                   # 增益小于阈值则返回
    root.n_split = j                                           # 保存用于划分的特征

    values = np.unique(D[:, j])                                # 获取特征j的各个值
    for value in values:                                       # 遍历特征j的各个值
        index = np.where(D[:, j] == value)                     # 筛出当前值的下标
        n_set_ = n_set.copy()                                  # 复制候选特征集合
        n_set_.remove(j)                                       # 删除已选用的特征
        sub_tree = build_decision_tree(D[index], n_set_, e)    # 递归构建决策树
        root.children.append((value, sub_tree[0]))             # 保存子结点信息
        tree += sub_tree                                       # 并入生成的子树

    return tree                                                # 返回生成的决策树

def get_max_label(D):  # 获取当前数据集最多的类别
    labels, counts = np.unique(D[:, -1], return_counts=True)   # 获取类别及对应数量
    index = np.argmax(counts)                                  # 数量最多的类的下标

    return labels[index]                                       # 返回数量最多的类别

def select_split_feature(D, n_set):  # 选出用于划分的特征
    n_list = list(n_set)                                       # 将特征集合转换为列表
    GYA    = np.zeros(len(n_list))                             # 存储各个特征的信息增益
    HYA    = np.zeros(len(n_list))                             # 存储各个特征的条件熵
    HY     = get_HY(D)                                         # 存储当前数据集的信息熵

    for i, n in enumerate(n_list):                             # i为下标，n=n_list[i]
        values, counts = np.unique(D[:,n], return_counts=True) # 获取第n个特征各值及数量
        for j in range(len(values)):                           # 遍历该特征各个值
            index = np.where(D[:, n] == values[j])             # 从D中筛出各值下标
            HDi = get_HY(D[index])                             # 计算信息熵，依据式(9.1)
            HYA[i] += counts[j] / len(D) * HDi                 # 计算条件熵，依据式(9.3)
        GYA[i] = HY - HYA[i]                                   # 计算信息增益，依据式(9.4)

    index = np.argmax(GYA)                                     # 确定信息增益最大值的下标
    return n_list[index], GYA[index]                           # 返回选出的特征及增益值

def get_HY(D):  # 计算数据集D的信息熵，依据式(9.1)
    HY = 0.0                                                   # 存储信息熵
    labels, counts = np.unique(D[:, -1], return_counts=True)   # 获取各个类别及对应样本量
    for k in range(len(labels)):                               # 遍历所有类别
        pk = counts[k] / len(D)                                # 计算该类别的概率
        HY += -pk * np.log2(pk)                                # 计算信息熵

    return HY                                                  # 返回信息熵

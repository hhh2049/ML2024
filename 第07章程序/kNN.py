# encoding=utf-8
import numpy as np

class Node:  # kd树的结点类
    def __init__(self, x=None, depth=0, parent=None, left=None, right=None):
        self.x      = x        # 包含真实值的一个训练数据，长度为特征数+1
        self.depth  = depth    # 该结点在kd树中所处的高度，根结点高度为0
        self.parent = parent   # 该结点的父结点，用于回溯
        self.left   = left     # 该结点的左子结点，可能为空（None）
        self.right  = right    # 该结点的右子结点，可能为空（None）

def compute_distance(x1, x2):  # 计算两个数据的欧氏距离
    diff = x1 - x2                                    # 计算两个数据之差
    dist = np.sqrt(np.dot(diff, diff))                # 计算两个数据的欧氏距离
    return dist                                       # 返回两个数据的欧氏距离

def build_kd_tree(X, parent, depth):  # 递归构建kd树，参数：数据集合、父结点、当前高度
    tree = []                                         # 用一个列表存放构建的kd树
    m, n = X.shape                                    # 获取数据量和特征数
    if m == 1:                                        # 如果只有一个数据
        tree.append(Node(X[0], depth, parent))        # 构建一棵只有根节点的kd树
        return tree                                   # 返回kd树

    root_node = Node(depth=depth, parent=parent)      # 构建当前kd树的根结点
    left_X, right_X = [], []                          # 存放左右子树的训练数据
    j = depth % (n - 1)                               # 用第j列划分数据,X最后一列为真实值
    median = np.sort(X[:, j])[int(m/2)]               # 求第j列的中位数

    count = 0                                         # 用于计数中位数的个数
    for i in range(m):                                # 遍历当前所有数据
        if X[i, j] == median:                         # Xij等于中位数，分情况处理
            if count == 0:                            # 首次遇到中位数
                root_node.x = X[i]; count += 1        # 将数据放入根结点
            else:                                     # 此后再遇中位数,分情况处理
                if count % 2 == 1:                    # 奇数次遇到中位数
                    left_X.append(X[i]); count += 1   # 将数据放入左子树
                else:                                 # 偶数次遇到中位数
                    right_X.append(X[i]); count += 1  # 将数据放入右子树
        elif X[i, j] < median: left_X.append(X[i])    # 小于中位数,放入右子树
        else: right_X.append(X[i])                    # 大于中位数,放入左子树

    left_tree, right_tree = None, None                # 定义左、右子树
    if len(left_X) >= 1:                              # 左子树集合大于0,递归构建左子树
        left_tree = build_kd_tree(np.array(left_X), root_node, depth+1)
        root_node.left = left_tree[0]                 # 左子树根结点作为根结点左子结点
    if len(right_X) >= 1:                             # 右子树集合大于0,递归构建右子树
        right_tree = build_kd_tree(np.array(right_X), root_node, depth+1)
        root_node.right = right_tree[0]               # 右子树根结点作为根结点右子结点

    tree.append(root_node)                            # 当前根结点作为当前树第一个结点
    if left_tree is not None: tree += left_tree       # 左子树的结点添加到当前kd树
    if right_tree is not None: tree += right_tree     # 右子树的结点添加到当前kd树

    return tree                                       # 返回基于当前数据集X构建的kd树

def search_kd_tree(kd_tree_root, x, xnn, rnn, knn_set):  # 搜索kd树
    leaf_node = None                                      # 待查找的叶子结点
    node      = kd_tree_root                              # 用于查找叶子结点的当前结点
    j         = kd_tree_root.depth % len(x)               # 计算用于划分的第j列
    while node is not None:                               # 循环：寻找叶子结点
        if x[j] <= node.x[j]:                             # x[j]小于等于当前结点
            if node.left is not None: node = node.left    # 左子树若非空，则进入左子树
            else: leaf_node = node; break                 # 找到叶子结点，中止循环
        else:                                             # x[j]大于当前结点
            if node.right is not None: node = node.right  # 右子树若非空，则进入右子树
            else: leaf_node = node; break                 # 找到叶子结点，中止循环
        j = (j + 1) % len(x)                              # 计算下一个用于划分的第j列

    dist = compute_distance(leaf_node.x[:-1], x)          # 计算x到叶子结点的距离dist
    if dist < rnn and leaf_node not in knn_set:           # dist小于最小距离且不在集合中
        xnn, rnn = leaf_node, dist                        # 更新当前最近邻、当前最近距离

    parent_node = leaf_node.parent                        # 回溯：叶子结点父结点为父结点
    child_node  = leaf_node                               # 叶子结点作为父结点的子结点
    while parent_node is not None:                        # 循环：直到父结点为None
        dist = compute_distance(parent_node.x[:-1], x)    # 计算x到父结点的距离dist
        if dist < rnn and parent_node not in knn_set:     # dist小于最小距离且不在集合中
            xnn, rnn = parent_node, dist                  # 更新当前最近邻、当前最近距离

        j = parent_node.depth % len(x)                    # 计算用于划分的第j列
        if x[j] - rnn <= parent_node.x[j] <= x[j] + rnn:  # 判断超球体与超矩形是否相交
            if parent_node.left == child_node:            # 找到父结点的另一个子结点
                brother_node = parent_node.right
            else:
                brother_node = parent_node.left

            if brother_node is not None:                  # 兄弟结点非空，进入该区域搜索
                brother_node.parent = None                # 为递归能终止，需设父结点为空
                xnn, rnn = search_kd_tree(brother_node, x, xnn, rnn, knn_set)
                brother_node.parent = parent_node         # 兄弟结点的父结点恢复为原先值

        child_node  = parent_node                         # 将当前父结点作为子结点
        parent_node = parent_node.parent                  # 继续向上回溯和搜索

    return xnn, rnn                                       # 返回最近邻结点、最近距离

class kNN:  # k近邻的实现
    def __init__(self, X, y, k=5, algorithm="kd_tree", is_classify=True):
        self.X           = np.c_[X, y]                    # 真实值作为数据集的最后一列
        self.k           = k                              # k近邻的k值
        self.algorithm   = algorithm                      # 搜索最近邻的算法
        self.is_classify = is_classify                    # k近邻法用于分类还是回归
        self.kd_tree     = []                             # 生成的kd树，一个列表

    def find_knn_with_brute(self, x):  # 暴力搜索k近邻
        m = self.X.shape[0]                               # 获取训练数据集的数据量
        distances = np.zeros(m)                           # 存放x到各个数据的距离
        for i in range(m):                                # 遍历所有训练数据
            dist = compute_distance(self.X[i][:-1], x)    # 计算x到当前数据的距离
            distances[i] = dist                           # 保存距离

        knn_list = []                                     # 存放k个近邻和对应的距离
        index_sorted = np.argsort(distances)              # 对距离排序，返回下标
        index_k = index_sorted[:self.k]                   # 获取k个近邻的下标
        for i in range(self.k):                           # 循环k次
            j = index_k[i]                                # 获取第i个近邻(i<=k)下标
            knn_list.append((self.X[j], distances[j]))    # 保存当前近邻和对应的距离

        return knn_list                                   # 返回k个近邻和对应的距离

    def construct_kd_tree(self):  # 构建kd树
        self.kd_tree = build_kd_tree(self.X, None, 0)     # 递归构建kd树

    def find_knn_with_kd_tree(self, x):  # 通过kd树搜索k近邻
        knn_list = []                                     # 存放k个近邻和对应的距离
        knn_set = set()                                   # 存放当前已找到的近邻点
        for i in range(self.k):                           # 搜索k次，寻找k个近邻
            xnn, rnn = search_kd_tree(self.kd_tree[0], x, None, np.inf, knn_set)
            knn_set.add(xnn)                              # 将当前最近邻存入集合
            knn_list.append((xnn.x, rnn))                 # 将当前近邻、距离存入列表

        return knn_list                                   # 返回k个近邻和对应的距离

    def fit(self):  # 拟合数据，训练模型
        if self.algorithm == "kd_tree":                   # 如果搜索算法是kd树搜索
            self.construct_kd_tree()                      # 构建kd树
        elif self.algorithm == "brute":                   # 如果搜索算法是暴力搜索
            pass                                          # 啥也不做

    def predict_one(self, x):  # 预测一个数据
        if self.algorithm == "brute":                     # 如果算法为暴力搜索
            knn_list = self.find_knn_with_brute(x)        # 通过暴力搜索k个最近邻
        else:
            knn_list = self.find_knn_with_kd_tree(x)      # 通过kd树搜索k个最近邻

        if self.is_classify:                              # 如果是分类问题
            labels = np.zeros(self.k, dtype="int")        # 定义k个标签
            for i in range(self.k):                       # 遍历k个最近邻
                labels[i] = knn_list[i][0][-1]            # 获取标签值
            label_count = np.bincount(labels)             # 统计各个标签的数量
            y_hat = np.argmax(label_count)                # 获取数量最多的标签
        else:                                             # 如果是回归问题
            value_sum = 0                                 # k个近邻真实值之和
            for i in range(self.k):                       # 遍历k个最近邻
                value_sum += knn_list[i][0][-1]           # k个近邻真实值求和
            y_hat = value_sum / self.k                    # 将平均数作为预测值

        return y_hat                                      # 返回预测值

    def predict(self, X):  # 预测一个数据集
        m = X.shape[0]                                    # 获取数据集的数据量
        y_hat = np.zeros(m)                               # 用于存放m个预测值
        for i in range(m):                                # 遍历每个数据
            y_hat[i] = self.predict_one(X[i])             # 预测一个数据

        return y_hat                                      # 返回预测值向量

    def score(self, X, y):  # 计算分类准确率或回归的得分
        y_hat = self.predict(X)                           # 计算预测值

        if self.is_classify:                              # 如果是分类问题
            count = np.sum(y_hat == y)                    # 预测值与真实值相等的个数
            score = count / len(y)                        # 计算分类准确率
        else:                                             # 如果是回归问题
            diff = y - y_hat                              # 计算真实值与预测值之差
            mse = np.dot(diff, diff) / len(X)             # 计算MSE
            y_mean = np.mean(y)                           # 计算真实值的平均值
            diff = y - y_mean                             # 计算真实值与平均值之差
            var = np.dot(diff, diff) / len(X)             # 计算VAR
            score = 1.0 - mse / var                       # 计算回归得分

        return score                                      # 返回分类准确率或回归的得分

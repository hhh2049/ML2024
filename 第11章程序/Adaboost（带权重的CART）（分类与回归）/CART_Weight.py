# encoding=utf-8
import numpy as np

class Node:  # CART树的结点类
    def __init__(self, D=None, depth=None):
        self.D         = D       # 该结点包含的训练数据（含真实值）
        self.depth     = depth   # 该结点的深度
        self.y_hat     = None    # 该结点的预测值（分类标签或实数值）
        self.value     = None    # 该结点的gini指数或mse值
        self.n_split   = None    # 用于划分的特征的下标
        self.n_value   = None    # 用于划分的特征的值
        self.min_value = None    # 用于划分的最小gini指数或mse值
        self.left      = None    # 左子结点，叶子结点时为空（None）
        self.right     = None    # 右子结点，叶子结点时为空（None）

def build_dtree(D, depth, params):  # 构建CART分类与回归树
    if len(D) == 0: return None                              # 若D为空，则返回
    (is_classify, max_depth) = params[0:2]                   # 是否为分类，树的最大深度
    (min_sample_split, min_impurity_split) = params[2:4]     # 最小分裂样本数，最小分裂不纯度

    tree = []                                                # 用列表存储生成的决策树
    root = Node(D, depth)                                    # 定义当前树的根结点
    root.y_hat = get_predict_value(D, is_classify)           # 计算当前结点的预测值
    if is_classify: root.value = get_gini(D)                 # 若为分类计算基尼指数
    else: root.value = get_mse(D)                            # 若为回归计算mse值
    tree.append(root)                                        # 将根结点加入到树中

    if len(D) < min_sample_split: return tree                # 数据量小于最小分裂数
    if depth == max_depth: return tree                       # 树已达最大深度
    if is_classify:                                          # 如果是分类决策树
        if get_gini(D) <= min_impurity_split: return tree    # 基尼指数小于等于阈值
    else:                                                    # 如果是回归决策树
        if get_mse(D) <= min_impurity_split: return tree     # mse值小于等于阈值

    min, j, v = select_split_feature(D, is_classify)         # 选择划分特征
    root.min_value = min                                     # 保存最小基尼指数或mse值
    root.n_split, root.n_value = j, v                        # 保存用于划分的特征及值

    left_index   = np.where(D[:, j] <= v)                    # 小于等于划分值的数据下标
    right_index  = np.where(D[:, j] >  v)                    # 大于划分值的数据下标
    left_D, right_D = D[left_index], D[right_index]          # 划分左右子集

    if len(left_D) >= 1:                                     # 若左子集非空
        left_tree = build_dtree(left_D, depth + 1, params)   # 递归构建左子树
        root.left = left_tree[0]                             # 保存左子树的根结点
        tree += left_tree                                    # 保存左子树所有结点
    if len(right_D) >= 1:                                    # 若右子集非空
        right_tree = build_dtree(right_D, depth + 1, params) # 递归构建右子树
        root.right = right_tree[0]                           # 保存右子树的根结点
        tree += right_tree                                   # 保存左子树所有结点

    return tree                                              # 返回构建的决策树

def get_predict_value(D, is_classify=True):  # 计算当前结点的预测值（样本带权重）
    if is_classify:                                          # 如果是分类决策树
        # y = np.array([3, 2, 2, 1, 1, 2])                   # 示例数据
        # weight = np.array([0.4, 0.3, 0.3, 0.5, 0.5, 0.2])
        # labels = [1 2 3]
        # weight_count = [1.0 0.8 0.4]
        # labels[index] = 1
        y, weight    = D[:, -2], D[:, -1]                    # 获取真实值和样本权重
        labels       = np.unique(y)                          # 获取有多少个不同标签
        weight_count = np.zeros(len(labels))                 # 每一个标签有一个权重
        for i in range(len(y)):                              # 遍历当前所有真实值
            for j in range(len(labels)):                     # 遍历所有标签值
                if y[i] == labels[j]:                        # 当前数据属于哪个标签
                    weight_count[j] += weight[i]             # 增加标签对应的权重
        index = np.argmax(weight_count)                      # 获取最大权重值的下标
        return labels[index]                                 # 返回权重最大的标签
    else:                                                    # 如果是回归决策树
        # y = np.array([3.0, 2.0, 1.0])                      # 示例数据
        # weight = np.array([0.01, 0.02, 0.01])
        # normal_weight = [0.25 0.5 0.25]
        # np.dot(y, normal_weight) = 2.0
        y, weight     = D[:, -2], D[:, -1]                   # 获取真实值和样本权重
        normal_weight = weight / np.sum(weight)              # 归一化以后的权重
        return np.dot(y, normal_weight)                      # 返回真实值的均值（带权重）

def select_split_feature(D, is_classify):  # 选择用于划分的特征及相应的值
    m, n = D.shape[0], D.shape[1]-2                          # 获取数据量和特征数
    min_gini, min_mse = np.inf, np.inf                       # 存储最小基尼指数或mse值
    n_split,  n_value = -1, None                             # 存储用于划分的特征及值

    for j in range(n):                                       # 遍历所有特征
        fj = D[:, j].copy()                                  # 复制第j列特征的值
        fj = np.unique(fj)                                   # 特征值去重并排序
        for i in range(len(fj)-1):                           # 遍历切分点
            index1 = np.where(D[:, j] <= (fj[i]+fj[i+1])/2)  # 小于等于划分值的数据下标
            index2 = np.where(D[:, j] > (fj[i]+fj[i+1])/2)   # 大于划分值的数据下标
            D1, D2 = D[index1], D[index2]                    # 划分为两个数据集

            D_w = np.sum(D[:, -1])                           # 获取当前数据集的权重之和
            D1_w, D2_w = np.sum(D1[:,-1]), np.sum(D2[:,-1])  # 获取左右子集的权重之和

            if is_classify:                                  # 如果是分类决策树
                gini  = D1_w / D_w * get_gini(D1)            # 计算左子集基尼指数
                gini += D2_w / D_w * get_gini(D2)            # 计算累加右子集基尼指数
                if gini < min_gini:                          # 寻找最小基尼指数
                    min_gini = gini                          # 保存当前基尼指数
                    n_split, n_value = j, (fj[i]+fj[i+1])/2  # 保存当前切分特征及值
            else:                                            # 如果是回归决策树
                mse  = D1_w / D_w * get_mse(D1)              # 计算左子集mse值
                mse += D2_w / D_w * get_mse(D2)              # 计算右子集mse值
                if mse < min_mse:                            # 寻找最小mse值
                    min_mse = mse                            # 保存当前mse值
                    n_split, n_value = j, (fj[i]+fj[i+1])/2  # 保存当前切分特征及值

    if is_classify:                                          # 如果是分类决策树
        return min_gini, n_split, n_value                    # 返回相应值
    else:                                                    # 如果是回归决策树
        return min_mse,  n_split, n_value                    # 返回相应值

def get_gini(D):  # 计算数据集D的基尼指数（带权重）
    if len(D) == 0: return 0                                 # 若D为空，则返回0

    y, weight = D[:, -2], D[:, -1]                           # 获取真实值和样本权重
    labels = np.unique(y)                                    # 获取有多少个不同标签
    weight_count = np.zeros(len(labels))                     # 每一个标签有一个权重
    for i in range(len(y)):                                  # 遍历当前所有真实值
        for j in range(len(labels)):                         # 遍历所有标签值
            if y[i] == labels[j]:                            # 当前数据属于哪个标签
                weight_count[j] += weight[i]                 # 增加标签对应的权重

    p = weight_count / np.sum(weight_count)                  # 各个类别的权重归一化
    gini = 1.0                                               # 基尼指数初值
    for i in range(len(p)):                                  # 遍历各个类别
        gini = gini - p[i] * p[i]                            # 计算基尼指数
    return gini                                              # 返回基尼指数

def get_mse(D):  # 计算数据集D的均方误差（带权重）
    if len(D) == 0: return 0                                 # 若D为空，则返回0

    # y = np.array([3.0, 2.0, 1.0])                          # 示例数据
    # weight = np.array([0.01, 0.02, 0.01])
    # diff = [1.0 0.0 -1.0]
    # normal_weight = [0.25 0.5 0.25]
    # mse = 0.5
    y, weight = D[:, -2], D[:, -1]                           # 获取真实值和对应的权重
    normal_weight = weight / np.sum(weight)                  # 归一化权重
    diff = y - np.mean(y)                                    # 计算真实值与均值之差
    mse = 0.0                                                # 用于存放mse
    for i in range(len(y)):                                  # 遍历所有数据
        mse += diff[i] * diff[i] * normal_weight[i]          # 计算mse

    return mse                                               # 返回mse值

class CART:  # CART决策树的实现
    def __init__(self, X, y, is_classify=True, max_depth=None, sample_weight=None,
                 min_sample_split=2, min_impurity_split=0.0):
        if sample_weight is None:                            # 如果样本权重为空
            sample_weight = np.array([1/len(X)] * len(X))    # 所有样本权重相等
        else:                                                # 如果样本权重非空
            sum_weight = np.sum(sample_weight)               # 求样本权重之和
            sample_weight = sample_weight / sum_weight       # 样本权重归一化

        self.X                  = np.c_[X, y, sample_weight] # 真实值、权重并入数据集
        self.is_classify        = is_classify                # 是分类还是回归
        self.max_depth          = max_depth                  # 最大深度阈值
        self.min_sample_split   = min_sample_split           # 最小分裂阈值
        self.min_impurity_split = min_impurity_split         # 最小纯度阈值
        self.tree               = None                       # 存放生成的树

    def fit(self):  # 拟合数据，训练模型，递归构建分类与回归树
        is_clsf, max_dpth = self.is_classify, self.max_depth
        smp_spl, imp_splt = self.min_sample_split, self.min_impurity_split
        params = (is_clsf, max_dpth, smp_spl, imp_splt)
        self.tree = build_dtree(self.X, depth=0, params=params)

    def predict_one(self, x):  # 预测一个数据
        if self.tree is None: return None                    # 若树为空返回

        y_hat = None                                         # 存储预测值
        node  = self.tree[0]                                 # 获取树根结点
        while node is not None:                              # 若当前结点非空
            y_hat = node.y_hat                               # 获取当前结点预测值
            if node.left is None and node.right is None:     # 若为叶子结点
                break                                        # 结束预测
            if x[node.n_split] <= node.n_value:              # 小于等于划分值
                node = node.left                             # 进入左子树
            else:                                            # 大于等于划分值
                node = node.right                            # 进入右子树

        return y_hat                                         # 返回预测值

    def predict(self, X):  # 预测一个数据集
        y_hat = np.zeros(len(X))                             # 定义预测值向量
        for i in range(len(X)):                              # 遍历每个数据
            y_hat[i] = self.predict_one(X[i])                # 预测每个数据

        return y_hat                                         # 返回预测结果

    def score(self, X, y, weight=None):  # 计算分类或回归得分
        if weight is None:                                   # 如果样本权重为空
            weight = np.array([1/len(X)] * len(X))           # 所有样本权重相等
        else:                                                # 如果样本权重非空
            weight = weight / np.sum(weight)                 # 样本权重归一化

        y_hat = self.predict(X)                              # 计算预测值
        if self.is_classify:                                 # 如果是分类问题
            score = 0.0                                      # 用于存储分类得分
            for i in range(len(y)):                          # 遍历所有数据
                if y_hat[i] == y[i]:                         # 若真实值预测值相等
                    score += weight[i]                       # 计算得分（权重和为1）
        else:                                                # 如果是回归问题
            diff = y - y_hat                                 # 真实值与预测值之差
            mse = 0.0                                        # 用于存储均方误差
            for i in range(len(y)):                          # 遍历所有数据
                mse += diff[i] * diff[i] * weight[i]         # 计算mse
            y_mean = np.mean(y)                              # 真实值的平均值
            diff = y - y_mean                                # 真实值与平均值之差
            var = np.dot(diff, diff) / len(X)                # 计算var
            score = 1.0 - mse / var                          # 计算回归得分

        return score                                         # 返回得分

# encoding=utf-8
import numpy as np

class Node:  # CART树的结点类
    def __init__(self, D=None, depth=None):  # 初始化函数
        self.D         = D       # 该结点包含的训练数据（含一、二阶导数）
        self.depth     = depth   # 该结点的深度
        self.y_hat     = None    # 该结点的预测值
        self.value     = None    # 该结点的不纯度
        self.n_split   = None    # 用于划分的特征的下标
        self.n_value   = None    # 用于划分的特征的值
        self.min_value = None    # 用于划分的最小不纯度
        self.left      = None    # 左子结点，叶子结点时为空（None）
        self.right     = None    # 右子结点，叶子结点时为空（None）

def build_tree(D, depth, params):  # 构建CART回归树
    if len(D) == 0: return None                              # 若D为空，则返回
    (reg_lambda, gamma, max_depth) = params[0:3]             # 参数λ、γ，树的最大深度阈值
    (min_sample_split, min_child_weight) = params[3:5]       # 结点最小样本数、最小纯度

    tree = []                                                # 用列表存储生成的决策树
    root = Node(D, depth)                                    # 定义当前树的根结点
    root.y_hat = get_predict_value(D, reg_lambda)            # 计算当前结点的预测值
    root.value = get_impurity(D, reg_lambda, gamma)          # 计算当前结点的不纯度
    tree.append(root)                                        # 将根结点加入到当前树

    if depth == max_depth: return tree                       # 树已达最大深度
    if len(D) < min_sample_split: return tree                # 数据量小于最小分裂数
    if np.sum(D[:, -1]) <= min_child_weight: return tree     # 结点纯度小于等于阈值

    min, j, j_v = select_split_feature(D, reg_lambda, gamma) # 寻找最佳分裂特征及值
    if j == -1: return tree                                  # 若未找到则返回
    root.min_value, root.n_split, root.n_value = min, j, j_v # 保存最佳分裂特征及值

    left_index   = np.where(D[:, j] <= j_v)                  # 小于等于划分值的数据下标
    right_index  = np.where(D[:, j] >  j_v)                  # 大于划分值的数据下标
    left_D, right_D = D[left_index], D[right_index]          # 划分左右子集

    if len(left_D) >= 1:                                     # 若左子集非空
        left_tree = build_tree(left_D, depth + 1, params)    # 递归构建左子树
        root.left = left_tree[0]                             # 保存左子树的根结点
        tree += left_tree                                    # 保存左子树所有结点
    if len(right_D) >= 1:                                    # 若右子集非空
        right_tree = build_tree(right_D, depth + 1, params)  # 递归构建右子树
        root.right = right_tree[0]                           # 保存右子树的根结点
        tree += right_tree                                   # 保存左子树所有结点

    return tree                                              # 返回构建的决策树

def get_predict_value(D, reg_lambda):  # 计算当前结点的预测值
    Gj, Hj = np.sum(D[:, -2]), np.sum(D[:, -1])              # 获取一、二阶导数之和
    wj = -1.0 * Gj / (Hj + reg_lambda)                       # 计算结点的预测值

    return wj                                                # 返回结点的预测值

def get_impurity(D, reg_lambda, gamma):  # 计算一个结点的不纯度
    if len(D) == 0: return 0                                 # 若数据集D为空则返回0
    Gj, Hj = np.sum(D[:, -2]), np.sum(D[:, -1])              # 获取一、二阶导数之和
    impurity = -0.5 * Gj * Gj / (Hj + reg_lambda) + gamma    # 计算结点的不纯度

    return impurity                                          # 返回结点的不纯度

def select_split_feature(D, reg_lambda, gamma):  # 寻找用于划分的特征及相应的值
    max_gain, n_split, n_value = 0, -1, None                 # 最大增益、分裂特征及值

    for j in range(D.shape[1]-2):                            # 遍历所有特征
        fj = D[:, j].copy()                                  # 复制第j列特征的值
        fj = np.unique(fj)                                   # 去掉重复值并排序
        for i in range(len(fj)-1):                           # 遍历当前特征的切分点
            index1 = np.where(D[:, j] <= (fj[i]+fj[i+1])/2)  # 小于等于划分值的数据下标
            index2 = np.where(D[:, j] >  (fj[i]+fj[i+1])/2)  # 大于划分值的数据下标
            D1, D2 = D[index1], D[index2]                    # 划分为两个数据集

            Dim  = get_impurity(D, reg_lambda, gamma)        # 获取当前数据集的不纯度
            D1im = get_impurity(D1, reg_lambda, gamma)       # 获取左子集的不纯度
            D2im = get_impurity(D2, reg_lambda, gamma)       # 获取右子集的不纯度

            gain = -(D1im + D2im - Dim)                      # 计算不纯度的增益
            if gain > max_gain:                              # 当前增益大于最大增益
                max_gain = gain                              # 保存当前增益
                n_split, n_value = j, (fj[i]+fj[i+1])/2      # 保存划分特征及值

    return max_gain, n_split, n_value                        # 返回增益、分裂特征及值

class CART_XGBoost:  # CART_XGBoost决策树的实现
    def __init__(self, X, y, y_hat=None, objective="squarederror",
                 reg_lambda=1e-5, gamma=0.0, max_depth=None,
                 min_sample_split=2, min_child_weight=1.0):
        if objective == "squarederror":                      # 若目标函数为平方误差
            self.grad = y_hat - y                            # 计算一阶导数
            self.hess = np.array([1.0] * len(y))             # 计算二阶导数
        elif objective == "logistic":                        # 如果目标函数为交叉熵
            y_hat = 1.0 / (1.0 + np.exp(-y_hat))             # 计算逻辑斯蒂函数值
            self.grad = y_hat - y                            # 计算一阶导数
            self.hess = y_hat * (1.0 - y_hat)                # 计算二阶导数

        self.X                = np.c_[X,self.grad,self.hess] # 一二阶导数并入数据
        self.reg_lambda       = reg_lambda                   # 正则化参数λ
        self.gamma            = gamma                        # 正则化参数γ
        self.max_depth        = max_depth                    # 树的最大深度阈值
        self.min_sample_split = min_sample_split             # 结点最小分裂阈值
        self.min_child_weight = min_child_weight             # 结点最小纯度阈值
        self.tree             = None                         # 用于存放生成的树

    def fit(self):  # 拟合数据，训练模型，递归构建CART回归树
        lamda, gamma, max_depth = self.reg_lambda, self.gamma, self.max_depth
        min_smp_spl, min_cld_wet = self.min_sample_split, self.min_child_weight
        params = (lamda, gamma, max_depth, min_smp_spl, min_cld_wet)
        self.tree = build_tree(self.X, depth=0, params=params)

    def predict_one(self, x):  # 预测一个数据
        if self.tree is None: return None                    # 若树为空则返回

        y_hat = None                                         # 存储预测值
        node  = self.tree[0]                                 # 获取树根结点
        while node is not None:                              # 若当前结点非空
            y_hat = node.y_hat                               # 获取当前结点预测值
            if node.left is None and node.right is None:     # 若为叶子结点
                break                                        # 结束预测
            if x[node.n_split] <= node.n_value:              # 若小于等于划分值
                node = node.left                             # 则进入左子树
            else:                                            # 若大于划分值
                node = node.right                            # 则进入右子树

        return y_hat                                         # 返回预测值

    def predict(self, X):  # 预测一个数据集
        y_hat = np.zeros(len(X))                             # 定义预测值向量
        for i in range(len(X)):                              # 遍历每个数据
            y_hat[i] = self.predict_one(X[i])                # 预测一个数据

        return y_hat                                         # 返回预测结果

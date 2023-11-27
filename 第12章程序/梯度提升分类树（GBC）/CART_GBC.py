# encoding=utf-8
import numpy as np

class Node:  # CART树的结点类
    def __init__(self, D=None, depth=None):
        self.D         = D       # 该结点包含的训练数据（含真实值、上一轮预测值）
        self.depth     = depth   # 该结点的深度
        self.y_hat     = None    # 该结点的预测值（实数值）
        self.value     = None    # 该结点的mse值
        self.n_split   = None    # 用于划分的特征的下标
        self.n_value   = None    # 用于划分的特征的值
        self.min_value = None    # 用于划分的最小mse值
        self.left      = None    # 左子结点，叶子结点时为空（None）
        self.right     = None    # 右子结点，叶子结点时为空（None）

def build_tree(D, depth, params):  # 构建CART回归树
    if len(D) == 0: return None                              # 若D为空，则返回
    max_depth = params[0]                                    # 树的最大深度阈值
    min_sample_split, min_impurity_split = params[1:3]       # 最小分裂样本数，最小分裂不纯度

    tree = []                                                # 用列表存储生成的决策树
    root = Node(D, depth)                                    # 定义当前树的根结点
    root.y_hat = get_predict_value(D)                        # 计算当前结点的预测值
    root.value = get_mse(D)                                  # 计算mse值
    tree.append(root)                                        # 将根结点加入到当前树中

    if len(D) < min_sample_split: return tree                # 数据量小于最小分裂数
    if depth == max_depth: return tree                       # 树已达最大深度
    if get_mse(D) <= min_impurity_split: return tree         # mse值小于等于阈值

    min, j, j_v = select_split_feature(D)                    # 寻找最佳分裂特征及值
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

def get_predict_value(D):  # 计算当前结点的预测值
    y, y_hat = D[:, -2], D[:, -1]                            # 获取真实值、当前预测值
    diff = y - y_hat                                         # 真实值、当前预测值之差
    rmj = np.sum(y_hat) / (np.dot(diff, 1.0-diff) + 1e-8)    # 计算当前结点的预测值

    return rmj                                               # 返回当前结点的预测值

def select_split_feature(D):  # 选择用于划分的特征及相应的值
    min_mse, n_split, n_value = np.inf, -1, None             # 最小mse、划分特征及值

    for j in range(D.shape[1]-2):                            # 遍历所有特征
        fj = D[:, j].copy()                                  # 复制第j列特征的值
        fj = np.unique(fj)                                   # 去掉重复值并排序
        for i in range(len(fj)-1):                           # 遍历当前特征的切分点
            index1 = np.where(D[:, j] <= (fj[i]+fj[i+1])/2)  # 小于等于划分值的数据下标
            index2 = np.where(D[:, j] >  (fj[i]+fj[i+1])/2)  # 大于划分值的数据下标
            D1, D2 = D[index1], D[index2]                    # 划分为两个数据集

            mse  = len(D1) / len(D) * get_mse(D1)            # 计算左子集mse值
            mse += len(D2) / len(D) * get_mse(D2)            # 计算右子集mse值并相加
            if mse <= min_mse:                               # 寻找最小mse值
                min_mse = mse                                # 保存当前mse值
                n_split, n_value = j, (fj[i]+fj[i+1])/2      # 保存当前切分特征及值

    return min_mse, n_split, n_value                         # 返回相应值

def get_mse(D):  # 计算数据集D的均方误差（注意使用y_hat）
    if len(D) == 0: return 0                                 # 若D为空，则返回
    y_hat, mean = D[:, -1], np.mean(D[:, -1])                # 计算真实值的均值
    mse = np.dot(y_hat-mean, y_hat-mean) / len(y_hat)        # 计算mse值

    return mse                                               # 返回mse值

class CART_GBC:  # CART_GBC决策树的实现
    def __init__(self, X, y, y_hat, max_depth=3,
                 min_sample_split=2, min_impurity_split=0.0):
        self.X                  = np.c_[X, y, y_hat]         # 真实值、预测值并入训练数据
        self.max_depth          = max_depth                  # 树最大深度阈值
        self.min_sample_split   = min_sample_split           # 最小样本数阈值
        self.min_impurity_split = min_impurity_split         # 最小不纯度阈值
        self.tree               = None                       # 存放生成的树

    def fit(self):  # 拟合数据，训练模型，递归构建回归树
        params = (self.max_depth, self.min_sample_split, self.min_impurity_split)
        self.tree = build_tree(self.X, depth=0, params=params)

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
            else:                                            # 大于划分值
                node = node.right                            # 进入右子树

        return y_hat                                         # 返回预测值

    def predict(self, X):  # 预测一个数据集
        y_hat = np.zeros(len(X))                             # 定义预测值向量
        for i in range(len(X)):                              # 遍历每个数据
            y_hat[i] = self.predict_one(X[i])                # 预测一个数据

        return y_hat                                         # 返回预测结果

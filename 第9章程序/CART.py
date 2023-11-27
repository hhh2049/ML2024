# encoding=utf-8
import copy
import numpy as np

class Node:  # CART树的结点类
    def __init__(self, D=None, depth=None):
        self.D         = D       # 该结点包含的训练数据（含真实值）
        self.depth     = depth   # 该结点的深度
        self.y_hat     = None    # 该结点的预测值（分类标签或实数值）
        self.n_split   = None    # 用于划分的特征的下标
        self.n_value   = None    # 用于划分的特征的值
        self.min_value = None    # 用于划分的最小基尼指数或mse值
        self.left      = None    # 左子结点，叶子结点时为空（None）
        self.right     = None    # 右子结点，叶子结点时为空（None）

def build_decision_tree(D, depth, params):  # 构建CART分类与回归树
    if len(D) == 0: return None                              # 若D为空，则返回
    (is_classify, max_depth, min_sample_split, min_impurity_split) = params
    tree = []                                                # 用列表存储生成的决策树
    root = Node(D, depth)                                    # 定义当前树的根结点
    root.y_hat = get_predict_value(D, is_classify)           # 计算当前结点的预测值
    tree.append(root)                                        # 将根结点加入到当前树中

    if len(D) < min_sample_split: return tree                # 数据量小于最小分裂数
    if depth == max_depth: return tree                       # 树已达最大深度
    if is_classify:                                          # 如果是分类决策树
        if get_gini(D) <= min_impurity_split: return tree    # 基尼指数小于等于阈值
    else:                                                    # 如果是回归决策树
        if get_mse(D) <= min_impurity_split: return tree     # mse值小于等于阈值

    min_value, j, j_value = select_split_feature(D, is_classify)  # 选择用于划分的特征
    root.min_value = min_value                               # 保存最小基尼指数或mse值
    root.n_split, root.n_value = j, j_value                  # 保存用于划分的特征及值

    left_index   = np.where(D[:, j] <= j_value)              # 小于等于划分值的数据下标
    right_index  = np.where(D[:, j] >  j_value)              # 大于划分值的数据下标
    left_D, right_D = D[left_index], D[right_index]          # 划分左右子集

    if len(left_D) >= 1:                                     # 若左子集非空
        left_tree = build_decision_tree(left_D, depth+1, params)  # 递归构建左子树
        root.left = left_tree[0]                             # 保存左子树的根结点
        tree += left_tree                                    # 保存左子树所有结点
    if len(right_D) >= 1:                                    # 若右子集非空
        right_tree = build_decision_tree(right_D, depth+1, params) # 递归构建右子树
        root.right = right_tree[0]                           # 保存右子树的根结点
        tree += right_tree                                   # 保存左子树所有结点

    return tree                                              # 返回构建的决策树

def get_predict_value(D, is_classify=True):  # 计算当前结点的预测值
    if is_classify:                                          # 如果是分类决策树
        labels, counts = np.unique(D[:, -1], return_counts=True) # 获取类别及数量
        index = np.argmax(counts)                            # 样例最多类别的下标
        return labels[index]                                 # 返回最多类别的标签
    else:                                                    # 如果是回归决策树
        return np.mean(D[:, -1])                             # 返回真实值的均值

def select_split_feature(D, is_classify):  # 选择用于划分的特征及相应的值
    m, n = D.shape[0], D.shape[1]-1                          # 获取数据量和特征数
    min_gini, min_mse = np.inf, np.inf                       # 存储最小基尼指数或mse值
    n_split,  n_value = -1, None                             # 存储用于划分的特征及值

    for j in range(n):                                       # 遍历所有特征
        fj = D[:, j].copy()                                  # 复制第j列特征的值
        fj.sort()                                            # 对第j列的值从小到大排序
        for i in range(m-1):                                 # 遍历m-1个切分点
            index1 = np.where(D[:, j] <= (fj[i]+fj[i+1])/2)  # 小于等于划分值的数据下标
            index2 = np.where(D[:, j] > (fj[i]+fj[i+1])/2)   # 大于划分值的数据下标
            D1, D2 = D[index1], D[index2]                    # 划分为两个数据集

            if is_classify:                                  # 如果是分类决策树
                gini  = len(D1) / len(D) * get_gini(D1)      # 计算左子集基尼指数
                gini += len(D2) / len(D) * get_gini(D2)      # 计算累加右子集基尼指数
                if gini <= min_gini:                         # 寻找最小基尼指数
                    min_gini = gini                          # 保存当前基尼指数
                    n_split, n_value = j, (fj[i]+fj[i+1])/2  # 保存当前切分特征及值
            else:                                            # 如果是回归决策树
                mse  = len(D1) / len(D) * get_mse(D1)        # 计算左子集mse值
                mse += len(D2) / len(D) * get_mse(D2)        # 计算右子集mse值
                if mse <= min_mse:                           # 寻找最小mse值
                    min_mse = mse                            # 保存当前mse值
                    n_split, n_value = j, (fj[i]+fj[i+1])/2  # 保存当前切分特征及值

    if is_classify:                                          # 如果是分类决策树
        return min_gini, n_split, n_value                    # 返回相应值
    else:                                                    # 如果是回归决策树
        return min_mse,  n_split, n_value                    # 返回相应值

def get_gini(D):  # 计算数据集D的基尼指数，依据式(9.9)
    if len(D) == 0: return 0                                 # 若D为空，则返回

    gini = 1.0                                               # 存储基尼值
    labels, counts = np.unique(D[:, -1], return_counts=True) # 统计各个类别及相应数量
    for k in range(len(labels)):                             # 遍历各个类别
        pk = counts[k] / len(D)                              # 计算该类别比例
        gini = gini - pk * pk                                # 计算基尼指数

    return gini                                              # 返回基尼指数

def get_mse(D):  # 计算数据集D的mse值
    if len(D) == 0: return 0                                 # 若D为空，则返回

    mse = 0.0                                                # 存储mse值
    mean = np.mean(D[:, -1])                                 # 计算当前所有样本的均值
    for i in range(len(D)):                                  # 遍历D的各个数据
        mse += (D[i][-1] - mean) ** 2                        # 计算误差平方和
    mse = mse / len(D)                                       # 计算均值

    return mse                                               # 返回mse值

def min_cost_complexity_pruning(tree: [Node]):  # 代价复杂度剪枝算法
    if tree is None: return None                             # 如果树为空则返回
    sample_total = len(tree[0].D)                            # 数据集的样本总数
    impurity, _ = get_tree_impurity(sample_total, tree[0])   # 获取初始树的不纯度
    alphas, trees, impurities = [0.0], [tree], [impurity]    # 存储a值,树,不纯度

    current_tree = copy.deepcopy(tree)                       # 深拷贝初始决策树
    while len(current_tree) > 1:                             # 循环直到单结点树
        gt_list, inner_node_list = [], []                    # 存储gt值和内部结点
        node_list = [current_tree[0]]                        # 用于遍历树的列表
        while len(node_list) > 0:                            # 遍历当前树
            node = node_list.pop(0)                          # 弹出列表的首结点
            if node.left is None and node.right is None:     # 如果非内部结点
                continue                                     # 继续遍历树
            gini = get_gini(node.D)                          # 计算结点基尼指数
            ct = len(node.D) / sample_total * gini           # 计算ct值
            ctt, tt = get_tree_impurity(sample_total, node)  # 计算ctt、tt值
            gt  = (ct - ctt) / (tt - 1)                      # 计算gt值
            gt_list.append(gt)                               # 保存当前结点gt值
            inner_node_list.append(node)                     # 保存当前结点
            node_list.append(node.left)                      # 列表加入左子结点
            node_list.append(node.right)                     # 列表加入右子结点

        min_gt_index = np.argmin(np.array(gt_list))          # 获取最小gt值的下标
        alphas.append(gt_list[min_gt_index])                 # 保存最小gt值即α值
        root = inner_node_list[min_gt_index]                 # 最小gt值的所在结点
        new_tree = delete_tree_branch(current_tree, root)    # 删除子树
        trees.append(new_tree)                               # 保存当前树
        impurity, _ = get_tree_impurity(sample_total, new_tree[0]) # 获取树不纯度
        impurities.append(impurity)                          # 保存当前树的不纯度
        current_tree = copy.deepcopy(new_tree)               # 深度拷贝当前树

    return alphas, trees, impurities                         # 返回a值,树,不纯度列表

def get_tree_impurity(sample_total: int, root: Node):  # 计算当前树的不纯度和叶子数
    if sample_total <= 0 or root is None: return 0.0, 0      # 如果参数异常返回

    tree_impurity = 0.0                                      # 存放当前树的不纯度
    leaf_count    = 0                                        # 存放当前树的叶子数
    nodes         = [root]                                   # 用于遍历树的列表
    while len(nodes) > 0:                                    # 遍历当前树
        node = nodes.pop(0)                                  # 弹出列表的首结点
        if node.left is None and node.right is None:         # 如果是叶子结点
            gini = get_gini(node.D)                          # 计算基尼指数
            leaf_impurity = len(node.D)/sample_total * gini  # 计算叶子结点不纯度
            tree_impurity += leaf_impurity                   # 累加叶子结点不纯度
            leaf_count += 1                                  # 叶子结点计数
        else:                                                # 如果是内部结点
            nodes.append(node.left)                          # 左子结点加入列表
            nodes.append(node.right)                         # 右子结点加入类别

    return tree_impurity, leaf_count                         # 返回不纯度和叶子数

def delete_tree_branch(tree: [Node], root: Node):  # 对当前树进行剪枝
    nodes = [root]                                           # 存放待剪所有结点
    i = 0                                                    # 存放列表元素下标
    while i != len(nodes):                                   # 未到列表最后结点
        if nodes[i].left is not None:                        # 左子结点非空
            nodes.append(nodes[i].left)                      # 添加左子结点
        if nodes[i].right is not None:                       # 右子结点非空
            nodes.append(nodes[i].right)                     # 添加右子结点
        i += 1                                               # 指向列表下个元素

    for i in range(len(nodes)):                              # 遍历待剪枝子树
        if i == 0:                                           # 如果是根结点
            nodes[i].n_split = nodes[i].n_value = None       # 根结点变叶子结点
            nodes[i].min_value = None
            nodes[i].left = nodes[i].right = None
            continue
        tree.remove(nodes[i])                                # 从树中删除该结点

    return tree                                              # 返回剪枝后的树

class CART:  # CART分类与回归树的实现
    def __init__(self, X, y, is_classify=True,
                 max_depth=None, min_sample_split=2, min_impurity_split=0.0,
                 alpha=0.0):
        self.X                  = np.c_[X, y]                # 真实值并入数据集
        self.is_classify        = is_classify                # 指示分类还是回归
        self.max_depth          = max_depth                  # 最大深度阈值
        self.min_sample_split   = min_sample_split           # 最小分裂阈值
        self.min_impurity_split = min_impurity_split         # 最小纯度阈值
        self.alpha              = alpha                      # 用于剪枝的α
        self.tree               = None                       # 存放生成的树

    def fit(self):  # 拟合数据，训练模型，生成决策树
        is_clsf, max_dpth = self.is_classify, self.max_depth
        smp_split, imp_split = self.min_sample_split, self.min_impurity_split
        params = (is_clsf, max_dpth, smp_split, imp_split)
        self.tree = build_decision_tree(self.X, depth=0, params=params)

    def predict_one(self, x):  # 预测一个数据
        if self.tree is None: return None                    # 若树为空返回

        y_hat = None                                         # 存储预测值
        node  = self.tree[0]                                 # 获取树根结点
        while node is not None:                              # 若当前结点非空
            y_hat = node.y_hat                               # 获取预测值
            if node.left is None and node.right is None:     # 若为叶子结点
                break                                        # 结束预测
            if x[node.n_split] < node.n_value:               # 小于划分值
                node = node.left                             # 进入左子树
            else:                                            # 大于等于划分值
                node = node.right                            # 进入右子树

        return y_hat                                         # 返回预测值

    def predict(self, X):  # 预测一个数据集
        y_hat = np.zeros(len(X))                             # 定义预测值向量
        for i in range(len(X)):                              # 遍历每个数据
            y_hat[i] = self.predict_one(X[i])                # 预测一个数据

        return y_hat                                         # 返回预测结果

    def score(self, X, y):  # 计算分类或回归得分
        y_hat = self.predict(X)                              # 计算预测值

        if self.is_classify:                                 # 如果是分类问题
            count = np.sum(y_hat == y)                       # 预测正确的个数
            score = count / len(y)                           # 计算分类得分
        else:                                                # 如果是回归问题
            diff = y - y_hat                                 # 真实值与预测值之差
            mse = np.dot(diff, diff) / len(X)                # 计算MSE
            y_mean = np.mean(y)                              # 真实值的平均值
            diff = y - y_mean                                # 真实值与平均值之差
            var = np.dot(diff, diff) / len(X)                # 计算VAR
            score = 1.0 - mse / var                          # 计算回归得分

        return score                                         # 返回得分

import numpy as np
from graphviz import Digraph  # 从graphviz包中导入绘制有向图的类

def get_gini(D):  # 计算数据集D的基尼指数，依据式(9.9)
    if len(D) == 0: return 0

    gini = 1.0
    labels, counts = np.unique(D[:, -1], return_counts=True)
    for k in range(len(labels)):
        pk = counts[k] / len(D)
        gini = gini - pk * pk

    return gini

class DecisionTreePlotter:  # CART分类与回归树可视化类
    def __init__(self, tree, feature_names=None, label_names=None):
        self.tree          = tree                             # 自实现的CART类生成的决策树
        self.feature_names = feature_names                    # 数据集的特征名称
        self.label_names   = label_names                      # 数据集的分类标签名称
        self.y             = tree[0].D[:, -1]                 # 数据集的分类标签向量
        self.k             = len(np.unique(self.y))           # 数据集的类别数量
        self.graph         = Digraph("Decision Tree")         # 定义有向图绘制类对象

    def build(self, dt_node):  # 绘制决策树（递归函数）
        D = dt_node.D                                         # 获取当前结点中的数据
        
        if dt_node.left or dt_node.right:                     # 如果是内部结点
            if self.feature_names:                            # 如果特征名称非空
                label = self.feature_names[dt_node.n_split]   # 添加特征名称
            else:                                             # 如果未设置特征名称
                label = "X[" + str(dt_node.n_split) + "]"     # 添加特征下标
                
            label += " <= " + str(round(dt_node.n_value, 3))  # 添加划分值
            gini = get_gini(D)                                # 计算该结点的基尼指数
            label += "\n" + "gini = " + str(round(gini, 3))   # 添加基尼指数
            label += "\n" + "samples = " + str(len(D))        # 添加当前结点的样本数
            counts = []                                       # 存放当前结点各类别的样本数
            for i in range(self.k):                           # 遍历所有类别
                indexes = np.where(D[:, -1] == i)             # 筛选当前类别数据的下标
                counts.append(len(D[indexes]))                # 将当前类别样本数放入列表
            label += "\n" + "n = " + str(counts)              # 添加各类别样本数

            # 生成一个内部结点
            self.graph.node(str(id(dt_node)), label=label, shape="box", fontsize="20", fontname="SimSun")

            if dt_node.left is not None:                      # 如果左子结点非空
                self.build(dt_node.left)                      # 递归绘制左子树
                # 生成父结点到左子结点的边
                self.graph.edge(str(id(dt_node)), str(id(dt_node.left)), label="True", fontsize="20", fontname="SimSun")
            if dt_node.right is not None:                     # 如果右子结点非空
                self.build(dt_node.right)                     # 递归绘制右子树
                # 生成父结点到右子结点的边
                self.graph.edge(str(id(dt_node)), str(id(dt_node.right)), label="False",fontsize="20",fontname="SimSun")
        else:                                                 # 如果是叶子结点
            gini = get_gini(D)                                # 计算当前结点的基尼指数
            label = "gini = " + str(round(gini, 3))           # 添加当前结点的基尼指数
            label += "\n" + "samples = " + str(len(D))        # 添加当前结点的样本数
            counts = []                                       # 存放当前结点各类别的样本数
            for i in range(self.k):                           # 遍历所有类别
                counts.append(len(D[np.where(D[:,-1] == i)])) # 添加当前结点各类别的样本数
            label += "\n" + "n = " + str(counts)              # 添加各类别的样本数

            # 生成一个叶子结点
            self.graph.node(str(id(dt_node)), label=label, shape="", fontsize="20", fontname="SimSun")

    def plot(self):
        self.build(self.tree[0])                              # 绘制决策树
        self.graph.view()                                     # 显示决策树

    def save(self, filename):
        self.build(self.tree[0])                              # 绘制决策树
        self.graph.save(filename)                             # 保存决策树的dot代码

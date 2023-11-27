# encoding=utf-8
import numpy as np
from DecisionTree import DecisionTree  # 导入自实现的DecisionTree类
from graphviz import Digraph           # 从graphviz包中导入绘制有向图的类

class DecisionTreePlotter:  # 决策树可视化类
    def __init__(self, tree, feature_dict=None, label_dict=None):
        self.tree         = tree                                         # 自实现的决策树
        self.feature_dict = feature_dict                                 # 数据集的特征名称
        self.label_dict   = label_dict                                   # 数据集的类别名称
        self.graph        = Digraph("Decision Tree")                     # 定义有向图绘制类

    def build(self, dt_node):  # 递归绘制决策树
        if dt_node.children:                                             # 如果是内部结点
            if self.feature_dict:                                        # 如果特征名称非空
                label = self.feature_dict[dt_node.n_split]["name"]       # 获取特征中文名称
            else: label = "X[" + str(dt_node.n_split) + "]"              # 生成特征英文名称
            self.graph.node(str(id(dt_node)), label=label, shape="box",  # 设置结点参数
                            fontsize="20", fontname="SimSun")            # 生成内部结点

            for feature_value, dt_child in dt_node.children:             # 遍历子结点
                self.build(dt_child)                                     # 递归绘制决策树
                if self.feature_dict:                                    # 如果特征名称非空
                    feature = self.feature_dict[dt_node.n_split]         # 获取字典中的特征
                    label = feature["value_names"][feature_value]        # 获取特征的具体值
                else: label = str(feature_value)                         # 设置特征的英文值
                self.graph.edge(str(id(dt_node)), str(id(dt_child)),     # 设置边的参数
                        label=label, fontsize="20", fontname="SimSun")   # 生成连结结点的边
        else:                                                            # 若是叶子结点
            if self.label_dict: label = self.label_dict[dt_node.y_hat]   # 获取类别中文名称
            else: label = str(dt_node.y_hat)                             # 生成类别英文名称
            self.graph.node(str(id(dt_node)), label=label,               # 设置结点参数
                            shape="", fontsize="20", fontname="SimSun")  # 生成叶子结点

    def plot(self):
        self.build(self.tree[0])                    # 递归绘制决策树
        self.graph.view()                           # 以PDF文件形式显示决策树
        self.graph.save("D:/dtree.dot")             # 保存决策树的dot代码
        # dot -Tpng D:/dtree.dot -o D:/dtree.png    # 将dot代码转换为图像的命令

def main():
    X = np.array([[0, 1, 1, 1, 0],
                  [0, 1, 2, 1, 0],
                  [1, 1, 0, 1, 1],
                  [0, 0, 2, 0, 0],
                  [0, 1, 0, 0, 2],
                  [0, 1, 2, 0, 0],
                  [0, 1, 1, 1, 2],
                  [2, 1, 2, 1, 1],
                  [2, 0, 1, 0, 0],
                  [0, 1, 0, 1, 2],
                  [2, 1, 2, 1, 0],
                  [1, 1, 2, 1, 1]])                                  # 小丽相亲训练数据集
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])               # 训练数据的分类标签
    our_dt = DecisionTree(X, y)                                      # 定义自实现决策树对象
    our_dt.fit()                                                     # 构建决策树

    feature_dict = {                                                 # 特征的名称(字典形式)
        0: {"name": "外貌",
            "value_names": {0: "不帅", 1: "中等", 2: "帅"}},
        1: {"name": "家境",
            "value_names": {0: "一般", 1: "好"}},
        2: {"name": "身高",
            "value_names": {0: "矮", 1: "中等", 2: "高"}},
        3: {"name": "性格",
            "value_names": {0: "一般", 1: "好"}},
        4: {"name": "学历",
            "value_names": {0: "低", 1: "中等", 2: "高"}}
    }
    label_dict = {0: "不考虑", 1: "考虑"}                             # 类别的名称(字典形式)
    dtp = DecisionTreePlotter(our_dt.tree, feature_dict, label_dict) # 定义决策树绘制对象
    dtp.plot()                                                       # 显示和保存决策树

if __name__ == "__main__":
    main()

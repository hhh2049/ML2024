# encoding=utf-8
import graphviz
import numpy as np
from sklearn import datasets
from CART import CART, min_cost_complexity_pruning
from CART_Graphviz import DecisionTreePlotter
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

def main():
    # 对比自实现与官方库的代价复杂度剪枝算法
    np.set_printoptions(suppress=True, precision=4, linewidth=80)    # 控制输出格式
    iris_data = datasets.load_iris()                                 # 载入iris分类数据集
    X, y = iris_data.data, iris_data.target                          # 获取数据和分类标签

    our_dtc = CART(X, y, is_classify=True)                           # 定义自实现CART树
    our_dtc.fit()                                                    # 生成决策树
    dtp = DecisionTreePlotter(our_dtc.tree, None, None)              # 定义决策树绘制对象
    dtp.plot()                                                       # 方法一：显示pdf文件
    dtp.save("D:/our_iris_tree.dot")                                 # 方法二：输出dot文件
    "dot -Tpng D:/our_iris_tree.dot -o D:/our_iris_tree.png"         # dot文件转图像的命令

    skl_dtc = DecisionTreeClassifier(random_state=0, ccp_alpha=0.0)  # 定义官方库CART树
    skl_dtc.fit(X, y)                                                # 生成决策树
    dot_data = tree.export_graphviz(skl_dtc, out_file=None)          # 方法一：输出dot文件
    graph = graphviz.Source(dot_data)                                # 载入dot文件
    graph.render("D:/skl_iris_tree")                                 # dot文件转为pdf文件
    tree.export_graphviz(skl_dtc, "D:/skl_iris_tree.dot")            # 方法二：保存dot文件
    "dot -Tpng D:/skl_iris_tree.dot -o D:/skl_iris_tree.png"         # dot文件转图像的命令

    alphas, trees, imps = min_cost_complexity_pruning(our_dtc.tree)  # 执行自实现剪枝算法
    result_dict = skl_dtc.cost_complexity_pruning_path(X, y)         # 执行官方库剪枝算法
    print("our own trees count  = ", len(trees))                     # 打印生成了多少棵树
    print("sklearn trees count  = ", len(result_dict["ccp_alphas"])) # 打印生成了多少棵树
    print("our own CCP alpha    = ", np.array(alphas))               # 打印各树的alpha值
    print("sklearn CCP alpha    = ", result_dict["ccp_alphas"])      # 打印各树的alpha值
    print("our own CCP impurity = ", np.array(imps))                 # 打印各树的不纯度
    print("sklearn CCP impurity = ", result_dict["impurities"])      # 打印各树的不纯度

if __name__ == "__main__":
    main()

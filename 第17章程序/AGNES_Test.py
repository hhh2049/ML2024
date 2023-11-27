# encoding=utf-8
import numpy as np
from AGNES import AGNES
from sklearn.cluster import AgglomerativeClustering as skl_AGNES
from sklearn.metrics import adjusted_rand_score
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 样本量
    n_samples = 200
    # 双圆形数据集
    X1, y1 = datasets.make_circles(n_samples, factor=0.5, noise=0.05)
    # 双半月形数据集
    X2, y2 = datasets.make_moons(n_samples, noise=0.05)
    # 具有不同方差的高斯分布数据集
    std = [1.0, 2.5, 0.5]
    X3, y3 = datasets.make_blobs(n_samples, cluster_std=std, random_state=170)
    # 各向异性的高斯分布数据集
    X4, y4 = datasets.make_blobs(n_samples, random_state=170)
    X4 = np.dot(X4, [[0.6, -0.6], [-0.4, 0.8]])
    # 各向同性的高斯分布数据集
    X5, y5 = datasets.make_blobs(n_samples, random_state=8)
    # 无结构数据集
    X6, y6 = np.random.rand(n_samples, 2), None
    # 数据集和相应簇的数量
    data_cluster = [(X1, 2), (X2, 2), (X3, 3), (X4, 3), (X5, 3), (X6, 3)]

    our_label_list = []                                       # 自实现算法生成的簇标签
    skl_label_list = []                                       # 官方库算法生成的簇标签
    for data, cluster in data_cluster:                        # 遍历所有数据集（共6个）
        our_model = AGNES(data, cluster, linkage="single")    # 生成自实现的AGNES类对象
        our_model.fit()                                       # 执行训练
        our_label = our_model.labels_                         # 获取所有样本的簇标签
        our_label_list.append(our_label)                      # 保存簇标签

        skl_model = skl_AGNES(cluster, linkage="single")      # 生成官方库的AGNES类对象
        skl_model.fit(data)                                   # 执行训练
        skl_label = skl_model.labels_                         # 获取所有样本的簇标签
        skl_label_list.append(skl_label)                      # 保存簇标签

        score = adjusted_rand_score(our_label, skl_label)     # 比较两个分布的相似度
        print("ARI score =", score)                           # 打印两个分布的相似度

    label_list = [our_label_list, skl_label_list]             # 集合自实现和官方库的簇标签
    for i, label in enumerate(label_list):                    # 遍历自实现和官方库的簇标签
        plt.figure(figsize=(15, 8))                           # 生成图框，同时设置图框大小
        for j in range(len(data_cluster)):                    # 可视化聚类后形成的簇
            plt.subplot(2, 3, j + 1)                          # 绘制两行三列第几个图
            if i == 0:                                        # 如果是绘制自实现散点图
                plt.title("Our AGNES on Dataset %d" % (j+1))  # 设置标题
            else:                                             # 如果是绘制官方库散点图
                plt.title("SKL AGNES on Dataset %d" % (j+1))  # 设置标题
            x = data_cluster[j][0][:, 0]                      # 要显示的样本的横坐标
            y = data_cluster[j][0][:, 1]                      # 要显示的样本的纵坐标
            sns.scatterplot(x=x, y=y, hue=label[j], s=25,     # 绘制散点图
                            style=label[j], palette="Set1")   # 设置样本大小样式颜色
    plt.show()                                                # 显示散点图
    
if __name__ == '__main__':
    main()

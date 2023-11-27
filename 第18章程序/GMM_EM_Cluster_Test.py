# encoding=utf-8
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from GMM_EM_Cluster import GaussianMixture as Our_GMM
from sklearn.mixture import GaussianMixture as SKL_GMM

def main():
    n_samples = 200
    # 双圆形数据
    X1, y1 = datasets.make_circles(n_samples, factor=0.5, noise=0.05)
    # 双半月形数据
    X2, y2 = datasets.make_moons(n_samples, noise=0.05)
    # 具有不同方差的高斯分布数据
    std = [1.0, 2.5, 0.5]
    X3, y3 = datasets.make_blobs(n_samples, cluster_std=std, random_state=170)
    # 各向异性高斯分布数据
    X4, y4 = datasets.make_blobs(n_samples, random_state=170)
    X4 = np.dot(X4, [[0.6, -0.6], [-0.4, 0.8]])
    # 各向同性高斯分布数据
    X5, y5 = datasets.make_blobs(n_samples, random_state=8)
    # 无结构数据
    X6, y6 = np.random.rand(n_samples, 2), None
    # 数据集和簇数量
    data_cluster_list = [(X1, 2), (X2, 2), (X3, 3), (X4, 3), (X5, 3), (X6, 3)]

    our_label_list = []                                       # 自实现算法生成的簇标签
    skl_label_list = []                                       # 官方库算法生成的簇标签
    for data, cluster in data_cluster_list:                   # 遍历所有数据集（共6个）
        our_model = Our_GMM(data, None, n_components=cluster) # 生成自实现GMM类对象
        our_model.train()                                     # 执行训练
        w, Mu = our_model.init_w, our_model.init_Mu           # 保存初始参数w、Mu
        Sigma = our_model.init_Sigma                          # 保存初始参数Sigma
        our_label_list.append(our_model.predict(data))        # 保存簇标签

        skl_model = SKL_GMM(n_components=cluster,
                            weights_init=w, means_init=Mu,
                            precisions_init=Sigma)            # 生成官方库GMM类对象
        skl_model.fit(data)                                   # 执行训练
        skl_label_list.append(skl_model.predict(data))        # 保存簇标签

    label_list = [our_label_list, skl_label_list]             # 集合自实现和官方库的簇标签
    for i, label in enumerate(label_list):                    # 遍历自实现和官方库的簇标签
        plt.figure(figsize=(15, 8))                           # 生成图框，同时设置图框大小
        for j in range(len(data_cluster_list)):               # 可视化聚类后形成的簇
            plt.subplot(2, 3, j + 1)                          # 绘制两行三列第几个图
            if i == 0:                                        # 如果是绘制自实现散点图
                plt.title("Our GMM on Dataset %d" % (j+1))    # 设置标题
            else:                                             # 如果是绘制官方库散点图
                plt.title("SKL GMM on Dataset %d" % (j+1))    # 设置标题
            x = data_cluster_list[j][0][:, 0]                 # 要显示的样本的横坐标
            y = data_cluster_list[j][0][:, 1]                 # 要显示的样本的纵坐标
            sns.scatterplot(x=x, y=y, hue=label[j], s=25,     # 绘制散点图
                            style=label[j], palette="Set1")   # 设置样本大小样式颜色
    plt.show()                                                # 显示散点图

if __name__ == "__main__":
    main()

# encoding=utf-8
from Isomap import IsometricMapping
from sklearn import datasets
from sklearn import manifold
import matplotlib.pyplot as plt

def main():
    X, color = datasets.make_s_curve(500, random_state=0)    # 生成一个三维流形
    our_Isomap = IsometricMapping(X, n_neighbors=12)         # 定义自实现Isomap对象
    our_Z      = our_Isomap.fit_transform()                  # 获取将维后的数据
    skl_Isomap = manifold.Isomap(n_neighbors=12)             # 定义官方库Isomap对象
    skl_Z      = skl_Isomap.fit_transform(X)                 # 获取降维后的数据

    figure = plt.figure(figsize=(14, 6))                     # 设置图像的大小
    figure.suptitle("our Isomap vs skl Isomap")              # 设置图像的标题

    pcs = plt.cm.Spectral                                    # 为缩短代码长度
    ax = figure.add_subplot(1, 3, 1, projection="3d")        # 添加第一个子图
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=pcs) # 绘制三维散点图
    ax.view_init(4, -60)                                     # 设置观察的角度

    ax = figure.add_subplot(1, 3, 2)                         # 添加第二个子图
    ax.scatter(our_Z[:, 0], our_Z[:, 1], c=color, cmap=pcs)  # 绘制三维散点图
    ax.set_title("our Isomap")                               # 设置子图的标题
    ax.axis("tight")                                         # 设置坐标轴范围

    ax = figure.add_subplot(1, 3, 3)                         # 添加第三个子图
    ax.scatter(skl_Z[:, 0], skl_Z[:, 1], c=color, cmap=pcs)  # 绘制三维散点图
    ax.set_title("skl Isomap")                               # 设置子图的标题
    ax.axis("tight")                                         # 设置坐标轴范围

    plt.show()                                               # 显示图像

if __name__ == "__main__":
    main()

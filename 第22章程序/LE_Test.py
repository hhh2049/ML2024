# encoding=utf-8
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn import datasets
from LE import LaplacianEigenMaps as LE
from sklearn.manifold import SpectralEmbedding as SE

def main():
    X, color = datasets.make_s_curve(1000, random_state=0)   # 生成一个三维流形

    our_LE = LE(X, n_components=2, gamma=10.0)               # 定义自实现LE对象
    our_Z  = our_LE.fit_transform()                          # 获取降维后的数据
    skl_SE = SE(n_components=2, affinity="rbf", gamma=10.0)  # 定义官方库LE对象
    skl_Z  = skl_SE.fit_transform(X)                         # 获取降维后的数据

    figure = plt.figure(figsize=(16, 6))                     # 设置图像的大小
    figure.suptitle("our LE vs skl LE")                      # 设置图像的标题

    pcs = plt.cm.Spectral                                    # 为缩短代码长度
    ax = figure.add_subplot(1, 3, 1, projection="3d")        # 添加第一个子图
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=pcs) # 绘制三维散点图
    ax.view_init(4, -60)                                     # 设置观察的角度

    ax = figure.add_subplot(1, 3, 2)                         # 添加第二个子图
    ax.scatter(our_Z[:, 0], our_Z[:, 1], c=color, cmap=pcs)  # 绘制二维散点图
    ax.set_title("our LE")                                   # 设置子图的标题
    ax.axis("tight")                                         # 设置坐标轴范围

    ax = figure.add_subplot(1, 3, 3)                         # 添加第三个子图
    ax.scatter(skl_Z[:, 0], skl_Z[:, 1], c=color, cmap=pcs)  # 绘制二维散点图
    ax.set_title("skl LE")                                   # 设置子图的标题
    ax.axis("tight")                                         # 设置坐标轴范围
    formatter = ticker.FormatStrFormatter('%.3f')            # 保留三位小数
    ax.xaxis.set_major_formatter(formatter)                  # x轴保留三位小数
    ax.yaxis.set_major_formatter(formatter)                  # y轴保留三位小数
    plt.show()                                               # 显示图像

if __name__ == "__main__":
    main()

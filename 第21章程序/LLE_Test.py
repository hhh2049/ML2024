# encoding=utf-8
import matplotlib.pyplot as plt
from sklearn import datasets
from LLE import LocallyLinearEmbedding as OUR_LLE
from sklearn.manifold import LocallyLinearEmbedding as SKL_LLE

def main():
    X, color = datasets.make_s_curve(1000, random_state=0)   # 生成一个三维流形

    our_LLE = OUR_LLE(X, n_neighbors=12, n_components=2)     # 定义自实现LLE对象
    our_LLE.fit_transform()                                  # 执行降维操作
    our_Z = our_LLE.X_                                       # 获取将降维后的数据

    skl_LLE = SKL_LLE(n_neighbors=12, random_state=0)        # 定义官方库LLE对象
    skl_LLE.fit(X)                                           # 执行降维操作
    skl_Z = skl_LLE.transform(X)                             # 获取将降维后的数据

    figure = plt.figure(figsize=(14, 6))                     # 设置图像大小
    figure.suptitle("our LLE vs skl LLE")                    # 设置图像标题

    pcs = plt.cm.Spectral                                    # 为不同类别分配不同颜色
    ax = figure.add_subplot(1, 3, 1, projection="3d")        # 添加第一个子图
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=pcs) # 绘制三维散点图
    ax.view_init(4, -60)                                     # 设置观察的角度

    ax = figure.add_subplot(1, 3, 2)                         # 添加第二个子图
    ax.scatter(our_Z[:, 0], our_Z[:, 1], c=color, cmap=pcs)  # 绘制二维散点图
    ax.set_title("our LLE")                                  # 设置子图的标题
    ax.axis("tight")                                         # 设置坐标轴范围

    ax = figure.add_subplot(1, 3, 3)                         # 添加第三个子图
    ax.scatter(skl_Z [:, 0], skl_Z[:, 1], c=color, cmap=pcs) # 绘制二维散点图
    ax.set_title("skl LLE")                                  # 设置子图的标题
    ax.axis("tight")                                         # 设置坐标轴范围
    plt.show()                                               # 显示图像

if __name__ == "__main__":
    main()

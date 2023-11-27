# encoding=utf-8
import seaborn as sns
import matplotlib.pyplot as plt
from PCA import PCA
from sklearn import datasets, decomposition

def main():
    iris = datasets.load_iris()                                # 载入鸢尾花数据集
    X, y = iris.data, iris.target                              # 获取数据集和标签

    our_pca = PCA(X, n_components=2)                           # 定义自实现PCA类对象
    our_pca.fit_transform()                                    # 执行训练（降维）
    print("our PCA ratio:", our_pca.lambdas_ratio_)            # 打印前k个特征值占比

    skl_pca = decomposition.PCA(n_components=2)                # 定义官方库PCA类对象
    skl_pca_X_ = skl_pca.fit_transform(X)                      # 执行训练（降维）
    print("skl PCA ratio:", skl_pca.explained_variance_ratio_) # 打印前k个特征值占比

    data_list = [our_pca.X_, skl_pca_X_]                       # 构建降维后的数据集列表
    plt.figure(figsize=(12, 5))                                # 生成图框并设置图框大小
    for i, data in enumerate(data_list):                       # 遍历降维后的数据集列表
        plt.subplot(1, 2, i + 1)                               # 创建一行两列第i+1个图
        X0, X1 = data[:, 0], data[:, 1]                        # 获取第1列和第2列数据
        if i == 0:                                             # 如果是自实现PCA
            plt.title("Our PCA")                               # 设置标题
            sns.scatterplot(x=X0, y=-X1, s=25, hue=y,          # 设置散点图样式
                            style=y, palette="Set1")           # 绘制散点图
        else:                                                  # 如果是官方库PCA
            plt.title("SKL PCA")                               # 设置标题
            sns.scatterplot(x=X0, y=X1, s=25,                  # 设置散点图样式
                            hue=y, style=y, palette="Set1")    # 绘制散点图
    plt.show()                                                 # 显示散点图

if __name__ == "__main__":
    main()

# encoding=utf-8
from SVD import SVD
from sklearn import datasets, decomposition
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    iris = datasets.load_iris()                                # 载入鸢尾花数据集
    X, y = iris.data, iris.target                              # 获取数据集和标签

    our_svd = SVD(X, n_components=2)                           # 定义自实现SVD类对象
    our_svd.fit_transform()                                    # 执行训练（降维）
    print("our SVD singular:", our_svd.sigma_[:2])             # 打印前K个奇异值

    skl_svd = decomposition.TruncatedSVD(n_components=2)       # 定义官方库SVD类对象
    skl_svd_X_ = skl_svd.fit_transform(X)                      # 执行训练（降维）
    print("skl SVD singular:", skl_svd.singular_values_)       # 打印前K个奇异值

    data_list = [our_svd.X_, skl_svd_X_]                       # 构建降维后的数据集列表
    plt.figure(figsize=(12, 5))                                # 生成图框并设置图框大小
    for i, data in enumerate(data_list):                       # 遍历降维后的数据集列表
        plt.subplot(1, 2, i + 1)                               # 创建一行两列第i+1个图
        X0, X1 = data[:, 0], data[:, 1]                        # 获取第1列和第2列数据
        if i == 0:                                             # 如果是自实现SVD
            plt.title("Our SVD")                               # 设置标题
            sns.scatterplot(x=X0, y=X1, s=25, hue=y,           # 设置散点图样式
                            style=y, palette="Set1")           # 绘制散点图
        else:                                                  # 如果是官方库SVD
            plt.title("SKL SVD")                               # 设置标题
            sns.scatterplot(x=X0, y=X1, s=25,                  # 设置散点图样式
                            hue=y, style=y, palette="Set1")    # 绘制散点图
    plt.show()                                                 # 显示散点图

if __name__ == "__main__":
    main()

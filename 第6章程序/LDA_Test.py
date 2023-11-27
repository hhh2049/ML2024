from LDA import LDA
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import NullFormatter

def main():
    # 测试一：分类，对比自实现LDA与官方库LDA的运行效果
    iris = datasets.load_iris()                                # 载入iris数据集
    X, y = iris.data, iris.target                              # 获取数据和标签
    Z = train_test_split(X, y, test_size=0.2, random_state=2)  # 划分训练集和测试集
    X_train, X_test, y_train, y_test = Z[0], Z[1], Z[2], Z[3]  # 训练集和测试集赋值

    our_own_lda = LDA(X_train, y_train, solver="eigen")        # 创建自实现LDA对象
    our_own_lda.fit()                                          # 训练模型
    our_score_train = our_own_lda.score(X_train, y_train)      # 训练集评分
    our_score_test  = our_own_lda.score(X_test, y_test)        # 测试集评分

    sklearn_lda = LinearDiscriminantAnalysis(solver="eigen")   # 创建官方库LDA对象
    sklearn_lda.fit(X_train, y_train)                          # 训练模型
    sk_score_train = sklearn_lda.score(X_train, y_train)       # 训练集评分
    sk_score_test  = sklearn_lda.score(X_test, y_test)         # 测试集评分

    print("our own LDA: train score = %f" % our_score_train)   # 打印自实现训练得分
    print("sklearn LDA: train score = %f" % sk_score_train)    # 打印官方库训练得分
    print("our own LDA: test  score = %f" % our_score_test)    # 打印自实现测试得分
    print("sklearn LDA: test  score = %f" % sk_score_test)     # 打印官方库测试得分

    # 测试二：降维，对iris数据集进行降维并图形化展示
    iris = datasets.load_iris()                       # 载入iris数据集
    X, y = iris.data, iris.target                     # 获取数据和标签

    our_own_lda = LDA(X, y, n_components=2)           # 创建自实现LDA对象
    our_own_lda.fit()                                 # 训练模型
    Z1 = our_own_lda.transform(X)                     # 对数据集X降维

    lda = LinearDiscriminantAnalysis(n_components=2)  # 创建官方库LDA对象
    lda.fit(X, y)                                     # 训练模型
    Z2 = lda.transform(X)                             # 对数据集X降维

    figure = plt.figure(figsize=(10, 5))              # 创建图像并设置大小
    plt.subplots_adjust(left=0.03, right=0.95,        # 调整图片边缘的距离
                        top=0.95, bottom=0.03)

    ax = figure.add_subplot(1, 2, 1)                  # 添加子图，1行2列第1个子图
    ax.set_title("our own LDA of IRIS dataset")       # 设置第1个子图标题
    ax.xaxis.set_major_formatter(NullFormatter())     # 不显示x轴坐标
    ax.yaxis.set_major_formatter(NullFormatter())     # 不显示y轴坐标
    sns.scatterplot(x=-Z1[:, 0], y=Z1[:, 1], hue=y,   # x、y为降维后的前两维数据
                    s=50, palette="Set1")             # 根据设置绘制散点图

    ax = figure.add_subplot(1, 2, 2)                  # 添加子图，1行2列第2个子图
    ax.set_title("sklearn LDA of IRIS dataset")       # 设置第2个子图标题
    ax.xaxis.set_major_formatter(NullFormatter())     # 不显示x轴坐标
    ax.yaxis.set_major_formatter(NullFormatter())     # 不显示y轴坐标
    sns.scatterplot(x=Z2[:, 0], y=Z2[:, 1], hue=y,    # x、y为降维后的前两维数据
                    s=50, palette="Set1")             # 根据设置绘制散点图
    plt.show()                                        # 显示图像

if __name__ == "__main__":
    main()
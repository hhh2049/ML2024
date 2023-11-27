# encoding=utf-8
from ml_tools import standardize_data
from kNN import kNN as Our_kNN
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as Skl_kNN
import matplotlib.pyplot as plt

def main():
    # 对比自实现与官方库kNN分类
    iris_data = datasets.load_iris()                               # 载入iris数据集
    X, y = iris_data.data, iris_data.target                        # 获取数据和标签
    # digits_data = datasets.load_digits()                         # 载入digits数据集
    # X, y = digits_data.data, digits_data.target                  # 获取数据和标签

    X, _ = standardize_data(X, None)                               # 训练数据标准化
    Z = train_test_split(X, y, test_size=0.2, random_state=1)      # 划分训练集和测试集
    X_train, X_test, y_train, y_test = Z[0], Z[1], Z[2], Z[3]      # 训练集和测试集赋值

    our_knn = Our_kNN(X_train, y_train, k=4, algorithm="brute")    # 创建自实现kNN对象
    our_knn.fit()                                                  # 训练模型
    our_score_test = our_knn.score(X_test, y_test)                 # 测试集评分
    print("our own kNN brute:   test score = %f" % our_score_test) # 自实现测试得分

    our_knn = Our_kNN(X_train, y_train, k=4, algorithm="kd_tree")  # 创建自实现kNN对象
    our_knn.fit()                                                  # 训练模型
    our_score_test = our_knn.score(X_test, y_test)                 # 测试集评分
    print("our own kNN kd_tree: test score = %f" % our_score_test) # 自实现测试得分

    skl_knn = Skl_kNN(n_neighbors=4, algorithm="kd_tree")          # 创建官方库kNN对象
    skl_knn.fit(X_train, y_train)                                  # 训练模型
    skl_score_test = skl_knn.score(X_test, y_test)                 # 测试集评分
    print("sklearn kNN kd_tree: test score = %f" % skl_score_test) # 官方库测试得分

    # 测试不同k对kNN近邻算法的影响
    k_list = [1, 3, 5, 7, 10, 13, 15, 18, 20, 30, 40, 50]          # 定义不同的k
    score_list = []                                                # 存放预测得分的列表
    for k in k_list:                                               # 使用不同的k
        our_knn = Our_kNN(X_train, y_train, k=k)                   # 定义自实现kNN对象
        our_knn.fit()                                              # 训练模型
        score = our_knn.score(X_test, y_test)                      # 计算预测得分
        score_list.append(score)                                   # 将得分加入列表

    figure = plt.figure()                                          # 定义图像
    axis = figure.add_subplot(1, 1, 1)                             # 添加1行1列第1个子图
    axis.plot(k_list, score_list, "s-")                            # 按照定义的格式画线
    axis.set_title("different k in kNN model on the iris dataset") # 设置图像标题
    axis.set_xlabel("k")                                           # 设置x轴标题
    axis.set_ylabel("score")                                       # 设置y轴标题
    plt.show()                                                     # 显示图像

if __name__ == "__main__":
    main()

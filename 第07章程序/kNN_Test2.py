# encoding=utf-8
from ml_tools import standardize_data
from kNN import kNN as Our_kNN
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor as Skl_kNN
from sklearn.linear_model import SGDRegressor

def main():
    # 对比我们自实现与官方库kNN回归、线性回归的运行效果
    boston_house = datasets.load_boston()                                            # 载入波士顿房价数据集
    X, y = boston_house.data, boston_house.target                                    # 获取数据和真实值

    X, _ = standardize_data(X, None)                                                 # 训练数据标准化
    Z = train_test_split(X, y, test_size=0.2, random_state=1)                        # 划分训练集和测试集
    X_train, X_test, y_train, y_test = Z[0], Z[1], Z[2], Z[3]                        # 训练集和测试集赋值

    our_knn = Our_kNN(X_train, y_train, k=1, algorithm="brute", is_classify=False)   # 创建自实现kNN对象
    our_knn.fit()                                                                    # 训练模型
    our_score_test = our_knn.score(X_test, y_test)                                   # 测试集评分
    print("our own kNN brute:    test score = %f" % our_score_test)                  # 自实现测试得分

    our_knn = Our_kNN(X_train, y_train, k=1, algorithm="kd_tree", is_classify=False) # 创建自实现kNN对象
    our_knn.fit()                                                                    # 训练模型
    our_score_test = our_knn.score(X_test, y_test)                                   # 测试集评分
    print("our own kNN kd_tree:  test score = %f" % our_score_test)                  # 自实现测试得分

    skl_knn = Skl_kNN(n_neighbors=1, algorithm="kd_tree")                            # 创建官方库kNN对象
    skl_knn.fit(X_train, y_train)                                                    # 训练模型
    skl_score_test = skl_knn.score(X_test, y_test)                                   # 测试集评分
    print("sklearn kNN kd_tree:  test score = %f" % skl_score_test)                  # 官方库测试得分

    skl_sgdRegressor = SGDRegressor(alpha=0.0)                                       # 使用官方库线性回归类
    skl_sgdRegressor.fit(X_train, y_train)                                           # 训练模型
    skl_score = skl_sgdRegressor.score(X_test, y_test)                               # 计算预测得分
    print("sklearn SGDRegressor: test score = %f\n" % skl_score)                     # 打印预测得分

if __name__ == "__main__":
    main()

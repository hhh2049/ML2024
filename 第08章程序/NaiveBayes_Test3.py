# encoding=utf-8
import numpy as np
from NaiveBayes import NaiveBayes
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split

def main():
    np.set_printoptions(suppress=True, precision=6)            # 设置打印六位小数

    iris = datasets.load_iris()                                # 载入鸢尾花数据集
    X, y = iris.data, iris.target                              # 获取数据集和标签
    # cancer = datasets.load_breast_cancer()                   # 载入乳腺癌数据集
    # X, y   = cancer.data, cancer.target                      # 获取数据集和标签
    # digits = datasets.load_digits()                          # 载入手写数字数据集
    # X, y   = digits.data, digits.target                      # 获取数据集和标签
    Z = train_test_split(X, y, test_size=0.2, random_state=1)  # 划分训练集和测试集
    (X_train, X_test, y_train, y_test) = Z                     # 训练集和测试集赋值

    our_nb = NaiveBayes(X_train, y_train, fit_type="GNB")      # 定义自实现类对象
    our_nb.fit()                                               # 训练模型
    # print(our_nb.class_count)                                # 各类别的样本量
    # print(np.exp(our_nb.class_log_prior))                    # 各类别的概率
    # print(our_nb.mu)                                         # 各类别数据的均值
    # print(our_nb.sigma)                                      # 各类别数据的方差
    our_score_test = our_nb.score(X_test, y_test)              # 计算测试得分
    print("our own GNB: test score = %f" % our_score_test)     # 打印测试得分
    print(our_nb.predict_proba([X_test[0]]))                   # 数据1属于各别类的概率
    print(our_nb.predict_log_proba([X_test[0]]), "\n")         # 数据1属于各类的对数概率

    skl_gnb = GaussianNB()                                     # 创建官方库类对象
    skl_gnb.fit(X_train, y_train)                              # 训练模型
    # print(skl_gnb.class_count_)                              # 各类别的样本量
    # print(skl_gnb.class_prior_)                              # 各类别的概率
    # print(skl_gnb.theta_)                                    # 各类别数据的均值
    # print(skl_gnb.var_)                                      # 各类别数据的方差
    skl_score_test = skl_gnb.score(X_test, y_test)             # 计算测试得分
    print("sklearn GNB: test score = %f" % skl_score_test)     # 打印测试得分
    print(skl_gnb.predict_proba([X_test[0]]))                  # 数据1属于各类的概率
    print(skl_gnb.predict_log_proba([X_test[0]]))              # 数据1属于各类的对数概率

if __name__ == "__main__":
    main()

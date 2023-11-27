# encoding=utf-8
import numpy as np
from NaiveBayes import NaiveBayes
from sklearn.naive_bayes import BernoulliNB

def main():
    X_train = np.array([[0, 1, 1, 1, 0],
                        [0, 1, 2, 1, 0],
                        [1, 1, 0, 1, 1],
                        [0, 0, 2, 0, 0],
                        [2, 1, 0, 0, 2],
                        [0, 1, 2, 0, 0],
                        [0, 1, 1, 1, 2],
                        [2, 1, 2, 1, 1],
                        [1, 0, 1, 0, 0],
                        [0, 1, 0, 1, 2],
                        [2, 1, 2, 1, 0],
                        [1, 1, 2, 1, 1]])                  # 小丽相亲训练数据集
    y_train = np.array([0, 0, 0, 0, 0, 0,
                        1, 1, 1, 1, 1, 1])                 # 训练数据的真实标签
    X_test = np.array([[2, 1, 1, 0, 2]])                   # 测试数据
    y_test = np.array([1])                                 # 测试数据的真实标签

    our_nb = NaiveBayes(X_train, y_train, fit_type="BNB")  # 定义伯努利朴素贝叶斯类对象
    our_nb.fit()                                           # 训练模型
    print("our own NaiveBayes test result:")               # 打印提示信息
    print(our_nb.class_count)                              # 各类别的样本量
    print(np.exp(our_nb.class_log_prior))                  # 各类别的概率
    print(our_nb.feature_count)                            # 各类别各特征值的数量
    print(np.exp(our_nb.feature_log_prob))                 # 各类别各特征值的概率
    print(our_nb.predict(X_test))                          # 预测数据所属的类
    print(our_nb.predict_proba(X_test))                    # 预测数据属于各类的概率
    print(our_nb.predict_log_proba(X_test), "\n")          # 预测数据属于各类的对数概率

    skl_nb = BernoulliNB(alpha=1.0)                        # 定义伯努利朴素贝叶斯类对象
    skl_nb.fit(X_train, y_train)                           # 训练模型
    print("sklearn NaiveBayes test result:")               # 打印提示信息
    print(skl_nb.class_count_)                             # 各类别的样本量
    print(np.exp(skl_nb.class_log_prior_))                 # 各类别的概率
    print(skl_nb.feature_count_)                           # 各类别各特征值的数量
    print(np.exp(skl_nb.feature_log_prob_))                # 各类别各特征值的概率
    print(skl_nb.predict(X_test))                          # 预测数据所属的类
    print(skl_nb.predict_proba(X_test))                    # 预测数据属于各类的概率
    print(skl_nb.predict_log_proba(X_test))                # 预测数据属于各类的对数概率

if __name__ == "__main__":
    main()

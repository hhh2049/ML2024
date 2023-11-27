# encoding=utf-8
import time
from CART_Weight import CART
from AdaBoostClassifier import AdaBoostClassifier
from sklearn import datasets, ensemble
from sklearn.model_selection import train_test_split

def main():
    # 导入官方库自带的鸢尾花数据集
    iris_data = datasets.load_iris()                          # 导入鸢尾花数据集
    X, y = iris_data.data[50:150], iris_data.target[50:150]   # 第二、三类线性不可分
    # X, y = iris_data.data[0:100], iris_data.target[0:100]   # 第一、二类线性可分
    y[0:50], y[50:100] = 1, -1                                # 分类标签设为1和-1

    # 划分训练集和测试集
    Z = train_test_split(X, y, test_size=0.3, random_state=0)
    (X_train, X_test, y_train, y_test) = Z

    # 使用自实现CART分类树
    start = time.time()
    our_cart = CART(X_train, y_train, max_depth=1)
    our_cart.fit()
    print("our own CART train score     = %.6f" % our_cart.score(X_train, y_train))
    print("our own CART test  score     = %.6f" % our_cart.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

    # 使用自实现Adaboost分类
    start = time.time()
    our_abc = AdaBoostClassifier(X_train, y_train, n_estimators=10)
    our_abc.fit()
    print("our own AdaBoost train score = %.6f" % our_abc.score(X_train, y_train))
    print("our own AdaBoost test  score = %.6f" % our_abc.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

    # 使用官方库Adaboost分类
    start = time.time()
    skl_abc = ensemble.AdaBoostClassifier(n_estimators=10)
    skl_abc.fit(X_train, y_train)
    print("sklearn AdaBoost train score = %.6f" % skl_abc.score(X_train, y_train))
    print("sklearn AdaBoost train score = %.6f" % skl_abc.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

if __name__ == "__main__":
    main()

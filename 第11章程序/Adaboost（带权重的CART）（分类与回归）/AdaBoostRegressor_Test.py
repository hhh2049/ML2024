# encoding=utf-8
import time
import numpy as np
from CART_Weight import CART
from AdaBoostRegressor import AdaBoostRegressor
from sklearn import datasets, ensemble
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

def main():
    # 回归数据集一：函数生成的样本数据集（100×4）
    X, y = make_regression(n_features=4, n_informative=2, random_state=1, shuffle=False)

    # 回归数据集二：自定义小数据集（10×1）
    # X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    # y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])

    # 回归数据集三：官方库自带的波士顿房价数据集（506×13）
    # boston_data = datasets.load_boston()
    # X, y = boston_data.data, boston_data.target

    # 回归数据集四：官方库自带的糖尿病数据集（442×10）
    # diabetes_data = datasets.load_diabetes()
    # X, y = diabetes_data.data, diabetes_data.target

    # 划分训练集和测试集
    Z = train_test_split(X, y, test_size=0.3, random_state=0)
    (X_train, X_test, y_train, y_test) = Z

    # 使用自实现CART回归树
    start = time.time()
    our_cart = CART(X_train, y_train, is_classify=False, max_depth=5)
    our_cart.fit()
    print("our own CART train score     = %.6f" % our_cart.score(X_train, y_train))
    print("our own CART test  score     = %.6f" % our_cart.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

    # 使用自实现Adaboost回归
    start = time.time()
    our_abr = AdaBoostRegressor(X_train, y_train, n_estimators=25)
    our_abr.fit()
    print("our own AdaBoost train score = %.6f" % our_abr.score(X_train, y_train))
    print("our own AdaBoost test  score = %.6f" % our_abr.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

    # 使用官方库Adaboost回归
    start = time.time()
    skl_abr = ensemble.AdaBoostRegressor(n_estimators=25)
    skl_abr.fit(X_train, y_train)
    print("sklearn AdaBoost train score = %.6f" % skl_abr.score(X_train, y_train))
    print("sklearn AdaBoost train score = %.6f" % skl_abr.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

if __name__ == "__main__":
    main()

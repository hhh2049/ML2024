# encoding=utf-8
import time
import numpy as np
from CART import CART
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from GradientBoostingRegressor import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor as SKL_GBR

def main():
    # 回归数据集一：官方库自带的波士顿房价数据集（506×13）
    boston_data = datasets.load_boston()
    X, y = boston_data.data, boston_data.target

    # 回归数据集二：官方库自带的糖尿病数据集（442×10）
    # diabetes_data = datasets.load_diabetes()
    # X, y = diabetes_data.data, diabetes_data.target

    # 划分训练集和测试集
    Z = train_test_split(X, y, test_size=0.3, random_state=0)
    (X_train, X_test, y_train, y_test) = Z

    # 使用自实现CART回归树
    start = time.time()
    our_cart = CART(X_train, y_train, is_classify=False, max_depth=1)
    our_cart.fit()
    print("our own CART train score = %.6f" % our_cart.score(X_train, y_train))
    print("our own CART test  score = %.6f" % our_cart.score(X_test, y_test))
    y_hat = our_cart.predict(X_test)
    print("mse  = %.4f" % (np.dot(y_test-y_hat, y_test-y_hat) / len(y_test)))
    end = time.time()
    print("time = %.2f\n" % (end - start))

    # 使用自实现梯度提升回归树
    start = time.time()
    our_gbr = GradientBoostingRegressor(X_train, y_train, n_estimators=20)
    our_gbr.fit()
    print("our own GBR train score  = %.6f" % our_gbr.score(X_train, y_train))
    print("our own GBR test  score  = %.6f" % our_gbr.score(X_test, y_test))
    print("mse  = %.4f" % our_gbr.mse(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

    # 使用官方库梯度提升回归树
    start = time.time()
    skl_gbr = SKL_GBR(n_estimators=20, learning_rate=1.0, max_depth=1)
    skl_gbr.fit(X_train, y_train)
    print("sklearn GBR train score  = %.6f" % skl_gbr.score(X_train, y_train))
    print("sklearn GBR train score  = %.6f" % skl_gbr.score(X_test, y_test))
    y_pred = skl_gbr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("mse  = %.4f" % mse)
    end = time.time()
    print("time = %.2f" % (end - start))

if __name__ == "__main__":
    main()

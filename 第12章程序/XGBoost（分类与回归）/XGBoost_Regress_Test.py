# encoding=utf-8
import time
from XGBoost import XGBoost
from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

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

    # 测试自实现XGBoost回归
    start = time.time()
    our_xgb = XGBoost(X_train, y_train, n_estimators=50, objective="squarederror")
    our_xgb.fit()
    print("our XGBoost regressor train score = %.6f" % our_xgb.score(X_train, y_train))
    print("our XGBoost regressor test  score = %.6f" % our_xgb.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

    # 测试官方库XGBoost回归
    start = time.time()
    ctq_xgb = XGBRegressor(n_estimators=50, objective="reg:squarederror")
    ctq_xgb.fit(X_train, y_train)
    print("ctq XGBoost regressor train score = %.6f" % ctq_xgb.score(X_train, y_train))
    print("ctq XGBoost regressor test  score = %.6f" % ctq_xgb.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

if __name__ == "__main__":
    main()

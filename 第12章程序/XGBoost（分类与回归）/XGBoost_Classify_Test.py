# encoding=utf-8
import time
from XGBoost import XGBoost
from sklearn import datasets
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def main():
    # 分类数据集一：官方库自带的鸢尾花数据集（150×4）
    iris_data = datasets.load_iris()                          # 导入鸢尾花数据集
    X, y = iris_data.data[50:150], iris_data.target[50:150]   # 第二三类线性不可分
    # X, y = iris_data.data[0:100], iris_data.target[0:100]   # 第一二类线性可分
    y[0:50], y[50:100] = 0, 1                                 # 分类标签设为0和1

    # 分类数据集二：官方库自带的乳腺癌数据集（569×30）
    # cancer_data = datasets.load_breast_cancer()
    # X, y = cancer_data.data, cancer_data.target

    # 划分训练集和测试集
    Z = train_test_split(X, y, test_size=0.3, random_state=0)
    (X_train, X_test, y_train, y_test) = Z

    # 测试自实现XGBoost分类
    start = time.time()
    our_xgb = XGBoost(X_train, y_train, n_estimators=50, objective="logistic")
    our_xgb.fit()
    print("our XGBoost classifier train score = %.6f" % our_xgb.score(X_train, y_train))
    print("our XGBoost classifier test  score = %.6f" % our_xgb.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

    # 测试官方库XGBoost分类
    start = time.time()
    ctq_xgb = XGBClassifier(n_estimators=50, objective="binary:logistic")
    ctq_xgb.fit(X_train, y_train)
    print("ctq XGBoost classifier train score = %.6f" % ctq_xgb.score(X_train, y_train))
    print("ctq XGBoost classifier test  score = %.6f" % ctq_xgb.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

if __name__ == "__main__":
    main()

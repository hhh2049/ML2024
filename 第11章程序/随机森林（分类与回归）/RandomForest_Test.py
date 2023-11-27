# encoding=utf-8
import time
from CART import CART
from RandomForest import RandomForest
from sklearn import datasets, ensemble
from sklearn.model_selection import train_test_split

def main():
    # 测试自实现和官方库随机森林（用于分类）
    # 分类数据集一：官方库自带的鸢尾花数据集
    iris_data = datasets.load_iris()
    X, y = iris_data.data, iris_data.target

    # 分类数据集二：官方库自带的乳腺癌数据集
    # cancer_data = datasets.load_breast_cancer()
    # X, y = cancer_data.data, cancer_data.target

    # 划分训练集和测试集
    Z = train_test_split(X, y, test_size=0.3, random_state=0)
    (X_train, X_test, y_train, y_test) = Z

    # 使用自实现CART分类树
    start = time.time()
    our_cart = CART(X_train, y_train, is_classify=True)
    our_cart.fit()
    print("our own CART train score = %.6f" % our_cart.score(X_train, y_train))
    print("our own CART test  score = %.6f" % our_cart.score(X_test, y_test))
    end = time.time()
    print("time = %.2f" % (end - start))

    # 使用自实现随机森林
    start = time.time()
    our_rf = RandomForest(X_train, y_train, is_classify=True, n_estimators=30)
    our_rf.fit()
    print("our own RF train score   = %.6f" % our_rf.score(X_train, y_train))
    print("our own RF train score   = %.6f" % our_rf.score(X_test, y_test))
    end = time.time()
    print("time = %.2f" % (end - start))

    # 使用官方库随机森林
    start = time.time()
    skl_rf = ensemble.RandomForestClassifier(n_estimators=30)
    skl_rf.fit(X_train, y_train)
    print("sklearn RF train score   = %.6f" % skl_rf.score(X_train, y_train))
    print("sklearn RF train score   = %.6f" % skl_rf.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

    # 测试自实现和官方库随机森林（用于回归）
    # 回归数据集一：官方库自带的波士顿房价数据集
    boston_data = datasets.load_boston()
    X, y = boston_data.data, boston_data.target

    # 划分训练集和测试集
    Z = train_test_split(X, y, test_size=0.3, random_state=0)
    (X_train, X_test, y_train, y_test) = Z

    # 使用自实现CART回归树
    start = time.time()
    our_cart = CART(X_train, y_train, is_classify=False)
    our_cart.fit()
    print("ourself CART train score = %.6f" % our_cart.score(X_train, y_train))
    print("ourself CART test  score = %.6f" % our_cart.score(X_test, y_test))
    end = time.time()
    print("time = %.2f" % (end - start))

    # 使用自实现随机森林
    start = time.time()
    our_rf = RandomForest(X_train, y_train, is_classify=False, n_estimators=10)
    our_rf.fit()
    print("ourself RF train score   = %.6f" % our_rf.score(X_train, y_train))
    print("ourself RF train score   = %.6f" % our_rf.score(X_test, y_test))
    end = time.time()
    print("time = %.2f" % (end - start))

    # 使用官方库随机森林
    start = time.time()
    skl_rf = ensemble.RandomForestRegressor(n_estimators=10)
    skl_rf.fit(X_train, y_train)
    print("sklearn RF train score   = %.6f" % skl_rf.score(X_train, y_train))
    print("sklearn RF train score   = %.6f" % skl_rf.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

if __name__ == "__main__":
    main()

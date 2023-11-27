# encoding=utf-8
import time
from CART import CART
from sklearn import datasets
from sklearn.model_selection import train_test_split
from GradientBoostingClassifier import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier as SKL_GBC

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

    # 测试自实现CART树分类
    start = time.time()
    our_cart = CART(X_train, y_train, max_depth=1)
    our_cart.fit()
    train_score = our_cart.score(X_train, y_train)
    test_score  = our_cart.score(X_test, y_test)
    print("our own CART train score = %.6f" % train_score)
    print("our own CART test  score = %.6f" % test_score)
    end = time.time()
    print("time = %.2f\n" % (end - start))

    # 测试自实现梯度提升分类树
    start = time.time()
    our_gbc = GradientBoostingClassifier(X_train, y_train, n_estimators=50)
    our_gbc.fit()
    train_score = our_gbc.score(X_train, y_train)
    test_score  = our_gbc.score(X_test, y_test)
    print("our own GBC train score  = %.6f" % train_score)
    print("our own GBC test  score  = %.6f" % test_score)
    end = time.time()
    print("time = %.2f\n" % (end - start))

    # 测试官方库梯度提升分类树
    start = time.time()
    skl_gbc = SKL_GBC(n_estimators=50, learning_rate=1.0, max_depth=1)
    skl_gbc.fit(X_train, y_train)
    train_score = skl_gbc.score(X_train, y_train)
    test_score  = skl_gbc.score(X_test, y_test)
    print("sklearn GBC train score  = %.6f" % train_score)
    print("sklearn GBC train score  = %.6f" % test_score)
    end = time.time()
    print("time = %.2f" % (end - start))

if __name__ == "__main__":
    main()

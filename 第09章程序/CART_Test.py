# encoding=utf-8
from CART import CART
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

def main():
    # 对比自实现与官方库CART分类与回归树
    iris_data = datasets.load_iris()                                # 载入iris分类数据集
    X, y = iris_data.data, iris_data.target                         # 获取数据和标签
    # boston_house = datasets.load_boston()                         # 载入房价回归数据集
    # X, y = boston_house.data, boston_house.target                 # 获取数据和真实值

    Z = train_test_split(X, y, test_size=0.3, random_state=1)       # 划分训练集和测试集
    (X_train, X_test, y_train, y_test) = Z                          # 训练集和测试集赋值

    our_dtc = CART(X_train, y_train, is_classify=True)              # 创建自实现类对象
    # our_dtc = CART(X_train, y_train, is_classify=False)           # 创建自实现类对象
    our_dtc.fit()                                                   # 训练模型
    our_score_test = our_dtc.score(X_test, y_test)                  # 测试集评分
    print("our own DecisionTree: test score = %f" % our_score_test) # 打印测试得分

    skl_dtc = DecisionTreeClassifier(random_state=0)                # 创建官方库类对象
    # skl_dtc = DecisionTreeRegressor(random_state=0)               # 创建官方库类对象
    skl_dtc.fit(X_train, y_train)                                   # 训练模型
    skl_score_test = skl_dtc.score(X_test, y_test)                  # 测试集评分
    print("sklearn DecisionTree: test score = %f" % skl_score_test) # 打印测试得分

if __name__ == "__main__":
    main()

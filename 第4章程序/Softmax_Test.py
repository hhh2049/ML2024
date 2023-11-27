# encoding=utf-8
from ml_tools import standardize_data
from Softmax import Softmax
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR

def main():
    # 对比自实现与官方库Softmax回归的运行效果
    iris = datasets.load_iris()                                   # 载入iris数据集
    X, y = iris.data, iris.target                                 # 获取数据和标签
    X, _ = standardize_data(X, None)                              # 训练数据标准化
    Z = train_test_split(X, y, test_size=0.2, random_state=1)     # 划分训练集和测试集
    X_train, X_test, y_train, y_test = Z[0], Z[1], Z[2], Z[3]     # 训练集和测试集赋值

    our_softmax = Softmax(X_train, y_train, K=3)                  # 创建自实现Softmax对象
    our_softmax.fit()                                             # 训练模型
    our_score_train = our_softmax.score(X_train, y_train)         # 计算训练集得分
    our_score_test = our_softmax.score(X_test, y_test)            # 计算测试集得分

    skl_softmax = LR(multi_class="multinomial", solver="sag")     # 创建官方库Softmax对象
    skl_softmax.fit(X_train, y_train)                             # 训练模型
    skl_score_train = skl_softmax.score(X_train, y_train)         # 计算训练集得分
    skl_score_test = skl_softmax.score(X_test, y_test)            # 计算测试集得分

    print("our own Softmax: train score = %f" % our_score_train)  # 自实现训练得分
    print("sklearn Softmax: train score = %f" % skl_score_train)  # 官方库训练得分
    print("our own Softmax: test  score = %f" % our_score_test)   # 自实现测试得分
    print("sklearn Softmax: test  score = %f" % skl_score_test)   # 官方库测试得分

if __name__ == "__main__":
    main()

# encoding=utf-8
import time
from ANN import ANN
from ml_tools import standardize_data
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def main():
    # 导入官方库自带的鸢尾花数据集（150×4）
    iris_data = datasets.load_iris()
    X, y = iris_data.data, iris_data.target

    # 导入官方库自带的乳腺癌数据集（569×30）
    # cancer_data = datasets.load_breast_cancer()
    # X, y = cancer_data.data, cancer_data.target

    # 训练数据标准化
    X, _ = standardize_data(X)

    # 划分训练集和测试集
    Z = train_test_split(X, y, test_size=0.3, random_state=0)
    (X_train, X_test, y_train, y_test) = Z

    # 使用自实现人工神经网络（用于分类）
    start = time.time()
    our_ann = ANN(X_train, y_train, is_classify=True, hidden_layer_sizes=(25,),
                  activation="logistic", learning_rate_init=0.1, max_iter=1000)
    our_ann.fit()
    end = time.time()
    print("our own ANN train score = %.6f" % our_ann.score(X_train, y_train))
    print("our own ANN train score = %.6f" % our_ann.score(X_test, y_test))
    print("time = %.2f\n" % (end - start))

    # 使用官方库多层感知机（用于分类）
    start = time.time()
    skl_mlp = MLPClassifier(hidden_layer_sizes=(25,),
                            activation="logistic", solver="sgd",
                            learning_rate_init=0.1, max_iter=5000,
                            batch_size=1, alpha=0.0, momentum=0.0)
    skl_mlp.fit(X_train, y_train)
    end = time.time()
    print("sklearn ANN train score = %.6f" % skl_mlp.score(X_train, y_train))
    print("sklearn ANN train score = %.6f" % skl_mlp.score(X_test, y_test))
    print("time = %.2f\n" % (end - start))

if __name__ == "__main__":
    main()

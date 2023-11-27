# encoding=utf-8
from ml_tools import normalize_data
from Logistic import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as sklearn_LR

def main():
    # 对比自实现与官方库逻辑回归（LogisticRegression）的运行效果
    iris_data = datasets.load_iris()                      # 载入iris数据集
    X, y = iris_data.data, iris_data.target               # 获取数据和标签
    # X, y = X[0:100], y[0:100]                           # 取前100条数据（线性可分）
    X, y = X[50:150], y[50:150]                           # 取后100条数据（线性不可分）
    y[0:50], y[50:100] = 0, 1                             # 修改数据标签为0和1
    X, _ = normalize_data(X, None)                        # 训练数据标准化
    X_train, X_test, y_train, y_test = train_test_split(  # 划分训练集和测试集
        X, y, test_size=0.2, random_state=1)              # 80%用于训练，20%用于测试

    our_lr = LogisticRegression(X_train, y_train)         # 创建自实现LR对象
    our_lr.fit()                                          # 训练模型
    our_score_train = our_lr.score(X_train, y_train)      # 训练集评分
    our_score_test  = our_lr.score(X_test, y_test)        # 测试集评分

    skl_lr = sklearn_LR()                                 # 创建官方库LR对象
    skl_lr.fit(X_train, y_train)                          # 训练模型
    skl_score_train = skl_lr.score(X_train, y_train)      # 训练集评分
    skl_score_test  = skl_lr.score(X_test, y_test)        # 测试集评分

    print("our own Logistic: train score = %f" % our_score_train)  # 自实现训练得分
    print("sklearn Logistic: train score = %f" % skl_score_train)  # 官方库训练得分
    print("our own Logistic: test  score = %f" % our_score_test)   # 自实现测试得分
    print("sklearn Logistic: test  score = %f" % skl_score_test)   # 官方库测试得分

if __name__ == "__main__":
    main()

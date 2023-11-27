# encoding=utf-8
import time
import numpy as np
from numpy.random import RandomState
from CART import CART
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, ensemble
from sklearn.model_selection import train_test_split

class AdaBoostClassifier:  # Adaboost分类（基于重采样）的实现
    def __init__(self, X, y, n_estimators=10, learning_rate=1.0):
        self.X            = X                                  # 训练数据集
        self.y            = y                                  # 训练数据的分类标签
        self.n_estimators = n_estimators                       # 弱学习器的数量
        self.eta          = learning_rate                      # 学习率
        self.e            = np.zeros(n_estimators)             # 各弱学习器的错误率
        self.a            = np.zeros(n_estimators)             # 各弱学习器的权重
        self.tree         = []                                 # 存放弱学习器的列表

    def bootstrap_sample(self, wt):  # 以概率分布进行抽样
        m = len(self.X)                                        # 训练数据集的样本数
        rs = RandomState()                                     # 定义随机状态对象
        index = rs.choice(np.arange(m), size=m, p=wt)          # 以概率分布进行抽样
        return index                                           # 返回抽样结果（下标）

    def fit(self):  # 拟合数据，训练模型
        m, eta = self.X.shape[0], self.eta                     # 训练数据集的样本数
        wt = np.array([1/m] * m)                               # 初始化各样本的权重

        for t in range(self.n_estimators):                     # 尝试生成各弱学习器
            index = self.bootstrap_sample(wt)                  # 以概率分布进行抽样
            X, y = self.X[index], self.y[index]                # 用下标筛选数据标签
            tree = CART(X, y, max_depth=None)                  # 使用自实现CART回归树
            tree.fit()                                         # 拟合数据，训练模型
            # tree = DecisionTreeClassifier(max_depth=None)    # 使用官方库CART回归树
            # tree.fit(X, y)                                   # 拟合数据，训练模型

            self.e[t] = 1 - tree.score(X, y)                   # 计算当前树的错误率
            if self.e[t] > 0.5:                                # 如果错误率超过0.5
                print("train wrong e > 0.5")                   # 打印错误信息
                break                                          # 提前中止训练
            b = self.e[t] + 1e-10                              # 防止下一行分母为0
            self.a[t] = 0.5 * np.log((1 - self.e[t]) / b)      # 计算弱学习器的权重

            j_set = set()                                      # 已更新权重的下标集合
            for i in range(len(X)):                            # 遍历当前训练数据
                y_hat = tree.predict_one(X[i])                 # 计算弱学习器预测值
                # y_hat = tree.predict([X[i]])[0]              # 使用官方库CART树预测
                j = index[i]                                   # 获取数据i的原始下标
                if j not in j_set:                             # 如果数据j未曾更新权重
                    wt[j] *= np.exp(-eta*self.a[t]*y[i]*y_hat) # 计算新的权重
                    j_set.add(j)                               # 将下标j加入集合
            wt = wt / np.sum(wt)                               # 新权重归一化
            self.tree.append(tree)                             # 将弱学习器加入列表

    def predict_one(self, x):  # 预测一个数据
        n_estimators = len(self.tree)                          # 获取弱学习器数量
        fx = 0.0                                               # 存放强学习器预测结果
        for t in range(n_estimators):                          # 遍历所有弱学习器
            y_hat = self.tree[t].predict_one(x)                # 使用弱学习器进行预测
            # y_hat = self.tree[t].predict([x])[0]             # 使用官方库CART树预测
            fx += self.eta * self.a[t] * y_hat                 # 计算强学习器预测结果

        return 1 if fx >= 0.0 else -1                          # 返回分类结果

    def predict(self, X): # 预测一个数据集
        y_hat = np.zeros(len(X))                               # 定义预测值向量
        for i in range(len(X)):                                # 遍历每个数据
            y_hat[i] = self.predict_one(X[i])                  # 预测每个数据

        return y_hat                                           # 返回预测结果

    def score(self, X, y):  # 计算训练或预测得分
        y_hat = self.predict(X)                                # 获取数据集X的预测值
        count = np.sum(y_hat == y)                             # 预测值真实值相等的个数

        return count / len(y)                                  # 返回训练或预测得分

def main():
    # 使用官方库自带的鸢尾花数据集（150×4）
    iris_data = datasets.load_iris()                           # 导入鸢尾花数据集
    X, y = iris_data.data[50:150], iris_data.target[50:150]    # 第二、三类线性不可分
    # X, y = iris_data.data[0:100], iris_data.target[0:100]    # 第一、二类线性可分
    y[0:50], y[50:100] = 1, -1                                 # 分类标签设为1和-1

    # 划分训练集和测试集
    Z = train_test_split(X, y, test_size=0.3, random_state=0)
    (X_train, X_test, y_train, y_test) = Z

    # 使用自实现CART分类树
    start = time.time()
    our_cart = CART(X_train, y_train, max_depth=1)
    our_cart.fit()
    print("our own CART train score     = %.6f" % our_cart.score(X_train, y_train))
    print("our own CART test  score     = %.6f" % our_cart.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

    # 使用自实现Adaboost分类
    start = time.time()
    our_abc = AdaBoostClassifier(X_train, y_train, n_estimators=50, learning_rate=1.0)
    our_abc.fit()
    print("our own AdaBoost train score = %.6f" % our_abc.score(X_train, y_train))
    print("our own AdaBoost test  score = %.6f" % our_abc.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

    # 使用官方库Adaboost分类
    start = time.time()
    skl_abc = ensemble.AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    skl_abc.fit(X_train, y_train)
    print("sklearn AdaBoost train score = %.6f" % skl_abc.score(X_train, y_train))
    print("sklearn AdaBoost train score = %.6f" % skl_abc.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

if __name__ == "__main__":
    main()

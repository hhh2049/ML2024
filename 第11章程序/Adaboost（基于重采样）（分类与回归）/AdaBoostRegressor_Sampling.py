# encoding=utf-8
import time
import numpy as np
from numpy.random import RandomState
from CART import CART
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets, ensemble
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

class AdaBoostRegressor:  # Adaboost回归（基于重采样）的实现
    def __init__(self, X, y, n_estimators=25, learning_rate=1.0, loss="linear"):
        self.X            = X                               # 训练数据集
        self.y            = y                               # 训练数据的真实值
        self.n_estimators = n_estimators                    # 弱学习器的数量
        self.eta          = learning_rate                   # 学习率
        self.loss         = loss                            # 损失函数类型
        self.e            = np.zeros(n_estimators)          # 各弱学习器的错误率
        self.w            = np.zeros(n_estimators)          # 各弱学习器的权重
        self.tree         = []                              # 存放弱学习器的列表

    def bootstrap_sample(self, wt): # 以概率分布进行抽样
        m  = len(self.X)                                    # 训练数据集的样本数
        rs = RandomState()                                  # 定义随机状态对象
        index = rs.choice(np.arange(m), size=m, p=wt)       # 以概率分布进行抽样
        return index                                        # 返回抽样结果（下标）

    def fit(self):  # 拟合数据，训练模型
        m, eta = self.X.shape[0], self.eta                  # 训练数据集的样本数
        wt = np.array([1/m] * m)                            # 初始化各样本的权重

        for t in range(self.n_estimators):                  # 生成各个弱学习器
            index = self.bootstrap_sample(wt)               # 以概率分布进行抽样
            X, y = self.X[index], self.y[index]             # 用下标筛选数据和真实值
            tree = CART(X,y,is_classify=False, max_depth=3) # 使用自实现CART回归树
            tree.fit()                                      # 拟合数据，训练模型
            # tree = DecisionTreeRegressor(max_depth=3)     # 使用官方库CART回归树
            # tree.fit(X, y)                                # 拟合数据，训练模型
            self.tree.append(tree)                          # 将弱学习器加入列表

            y_hat_t = tree.predict(X)                       # 使用当前树进行预测
            abs_diff = np.abs(y - y_hat_t)                  # 计算各数据误差的绝对值
            Et = np.max(abs_diff)                           # 获取误差绝对值的最大者
            if Et == 0:                                     # 如果误差最大值为0
                self.e[t], self.w[t] = 0.0, 1.0             # 错误率为0，权重为1
                print("Et is 0")                            # 打印提示信息
                break                                       # 完美拟合数据，提前中止
            eti = abs_diff / Et                             # 计算各数据的错误率
            if self.loss == "square":                       # 如果为平方误差
                eti = eti ** 2                              # 计算平方误差
            elif self.loss == "exponential":                # 如果为指数误差
                eti = 1.0 - np.exp(-eti)                    # 计算指数误差
            self.e[t] = np.dot(wt[index], eti)              # 计算当前树的错误率
            at = self.e[t] / (1 - self.e[t])                # 计算at
            self.w[t] = self.eta * np.log(1 / at)           # 计算当前学习器的权重

            j_set = set()                                   # 已更新权重的下标集合
            for i in range(m):                              # 遍历当前训练数据
                j = index[i]                                # 获取数据i的原始下标
                if j not in j_set:                          # 如果数据j未曾更新权重
                    wt[j] *= np.power(at, (1-eti[i]) * eta) # 计算新的权重
                    j_set.add(j)                            # 将下标j加入集合
            wt = wt / np.sum(wt)                            # 新权重归一化

    def predict_one(self, x):  # 预测一个数据
        n_estimators = len(self.tree)                       # 获取弱学习器数量
        if n_estimators == 0: return None                   # 弱学习器数量为0则返回
        self.w = self.w[:n_estimators]                      # 获取有效的权重

        y_hat = np.zeros(n_estimators)                      # 存放各弱学习器预测结果
        for i in range(n_estimators):                       # 遍历各个弱学习器
            y_hat[i] = self.tree[i].predict_one(x)          # 使用弱学习器进行预测
            # y_hat[i] = self.tree[i].predict([x])[0]       # 使用官方库CART树预测

        sorted_index = np.argsort(y_hat)                    # 获取排序后的预测值下标
        smallest_index = 0                                  # 最小的满足不等式的下标
        sum_w = 0.0                                         # 用于存放弱学习器权重和
        threshold = 0.5 * np.sum(self.w)                    # 权重和阈值
        for i in range(n_estimators):                       # 遍历各个弱学习器
            smallest_index = sorted_index[i]                # 获取当前权重下标
            sum_w += self.w[smallest_index]                 # 计算权重之和
            if sum_w >= threshold:                          # 如果权重和超过阈值
                break                                       # 找到目标下标，中止循环
        return y_hat[smallest_index]                        # 返回预测结果

    def predict(self, X):  # 预测一个数据集
        y_hat = np.zeros(len(X))                            # 定义预测值向量

        for i in range(len(X)):                             # 遍历每个数据
            y_hat[i] = self.predict_one(X[i])               # 预测每个数据

        return y_hat                                        # 返回预测结果

    def score(self, X, y):  # 计算训练或预测得分
        y_hat = self.predict(X)                             # 获取数据集X的预测值

        diff = y - y_hat                                    # 真实值与预测值之差
        mse = np.dot(diff, diff) / len(X)                   # 计算MSE
        y_mean = np.mean(y)                                 # 真实值的平均值
        diff = y - y_mean                                   # 真实值与平均值之差
        var = np.dot(diff, diff) / len(X)                   # 计算VAR
        score = 1.0 - mse / var                             # 计算回归得分

        return score                                        # 返回回归得分

def main():
    # 回归数据集一：函数生成的样本数据集（100×4）
    X, y = make_regression(n_features=4, n_informative=2, random_state=1, shuffle=False)

    # 回归数据集二：自定义小数据集（10×1）
    # X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    # y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])

    # 回归数据集三：官方库自带的波士顿房价数据集（506×13）
    # boston_data = datasets.load_boston()
    # X, y = boston_data.data, boston_data.target

    # 回归数据集四：官方库自带的糖尿病数据集（442×10）
    # diabetes_data = datasets.load_diabetes()
    # X, y = diabetes_data.data, diabetes_data.target

    # 划分训练集和测试集
    Z = train_test_split(X, y, test_size=0.3, random_state=0)
    (X_train, X_test, y_train, y_test) = Z

    # 使用自实现CART回归树
    start = time.time()
    our_cart = CART(X_train, y_train, is_classify=False, max_depth=3)
    our_cart.fit()
    print("our own CART train score     = %.6f" % our_cart.score(X_train, y_train))
    print("our own CART test  score     = %.6f" % our_cart.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

    # 使用自实现Adaboost回归
    start = time.time()
    our_abr = AdaBoostRegressor(X_train, y_train, n_estimators=25, learning_rate=1.0)
    our_abr.fit()
    print("our own AdaBoost train score = %.6f" % our_abr.score(X_train, y_train))
    print("our own AdaBoost test  score = %.6f" % our_abr.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

    # 使用官方库Adaboost回归
    start = time.time()
    sk_abr = ensemble.AdaBoostRegressor(n_estimators=25, learning_rate=1.0)
    sk_abr.fit(X_train, y_train)
    print("sklearn AdaBoost train score = %.6f" % sk_abr.score(X_train, y_train))
    print("sklearn AdaBoost train score = %.6f" % sk_abr.score(X_test, y_test))
    end = time.time()
    print("time = %.2f\n" % (end - start))

if __name__ == "__main__":
    main()

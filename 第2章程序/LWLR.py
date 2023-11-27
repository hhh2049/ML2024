# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt

class LWLR:  # 局部加权线性回归(Local Weights Linear Regression)的实现
    def __init__(self, X, y, k=1.0):
        self.X         = X                             # 训练数据集X(m×n)
        self.y         = y                             # 训练数据的真实值y(m,)
        self.k         = k                             # 高斯核函数的参数k
        self.m, self.n = X.shape                       # 获取数据集的数据量m和特征数n

    def fit(self):                                     # 拟合数据，训练模型
        pass                                           # 无需训练，直接预测

    def predict(self, X):                              # 计算数据集X的预测值
        m = X.shape[0]                                 # 获取X中的数据量
        y_hat = np.zeros(m, )                          # 定义预测值列表

        for i in range(m):                             # 对X中的每个数据进行预测
            y_hat[i] = self.predict_one_data(X[i])     # 计算并保存一个预测值

        return y_hat                                   # 返回预测值列表

    def predict_one_data(self, test_x):                # 计算一个数据的预测值
        W = np.eye(self.m)                             # 定义权重矩阵(对角矩阵)
        for i in range(self.m):                        # 计算权重矩阵
            diff = self.X[i] - test_x                  # 计算两个数据之差
            distance = np.dot(diff, diff)              # 计算两个数据距离
            scope = -2.0 * self.k * self.k             # 邻近点的范围参数
            W[i][i] = np.exp(distance / scope)         # 计算权重

        XTWX  = np.dot(np.dot(self.X.T, W), self.X)    # 计算XTWX
        XTWX  = XTWX + np.eye(self.n) * 1e-8           # 添加正则项
        XTWX_ = np.linalg.inv(XTWX)                    # 计算XTWX的逆矩阵
        XTWY  = np.dot(np.dot(self.X.T, W), self.y)    # 计算XTWY
        theta = np.dot(XTWX_, XTWY)                    # 计算theta

        return np.dot(test_x, theta)                   # 返回预测值

def main():
    x = np.linspace(0.0, 1.0, 150)                     # 生成测试数据
    fx = x * 2 + 0.5 + 0.1 * np.sin(x * 50)            # 计算真实值
    y_test = fx + np.random.normal(0, 0.03, len(x))    # 在真实值中加入噪声
    X_test = np.c_[np.ones(len(x)), x]                 # 添加值为1的列向量，消除偏置

    lwlr = LWLR(X_test, y_test, k=0.01)                # 定义模型 k=0.1,0.01,0.001
    y_predict = lwlr.predict(X_test)                   # 执行预测

    plt.plot(x, y_test, ".", markersize=3, color="b")  # 绘制测试数据点
    plt.plot(x, y_predict, "r")                        # 绘制拟合数据的曲线
    plt.show()                                         # 显示图像

if __name__ == "__main__":
    main()

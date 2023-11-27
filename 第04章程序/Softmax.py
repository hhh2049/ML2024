# encoding=utf-8
import numpy as np

def preprocess_label(y, K):  # 对真实分类标签y进行独热编码，K为分类数量
    m = len(y)               # 获取y的长度，即数据量
    Y = np.zeros((m, K))     # 定义Y矩阵(m×K)
    for i in range(m):       # 对y中每个分类标签进行独热编码
        j = y[i]             # 获取真实标签，真实分类标签为0、1、2...
        Y[i][j] = 1.0        # 将Y[i][j]置1.0，Y[i]行的其他位置默认为0

    return Y                 # 返回经独热编码的矩阵Y

class Softmax:  # Softmax回归算法的实现
    def __init__(self, X, y, K=3, eta=0.1, tol=1e-6, max_iter=1000):
        self.X         = np.c_[np.ones(len(X)), X]         # 训练数据集，X的维度m×(n+1)
        self.Y         = preprocess_label(y, K)            # 真实分类标签，Y的维度(m,K)
        self.K         = K                                 # 数据的类别数
        self.eta       = eta                               # 学习率
        self.tol       = tol                               # 训练终止的阈值
        self.max_iter  = max_iter                          # 最大的训练次数
        self.m, self.n = X.shape                           # 获取数据集的数据量和特征数
        self.W         = np.zeros((K, self.n + 1))         # 定义待学习的权重矩阵K×(n+1)

    def softmax(self, Xi):  # 计算一个数据的预测值，返回一个向量(K,)
        z = np.dot(self.W, Xi)                             # 计算每个数据的z值(K,)
        z = z - np.max(z)                                  # 防止计算np.exp(z)时溢出
        e_z = np.exp(z)                                    # 计算每个数据的exp(z)值(K,)
        y_hat = e_z / np.sum(e_z)                          # 计算每个数据的预测值(K,)

        return y_hat                                       # 返回预测值，如[0.1,0.3,0.6]

    def compute_loss(self, X, Y):  # 根据式（4.36），计算损失函数值
        m = X.shape[0]                                     # 获取X的长度，即数据量
        loss = 0.0                                         # 定义损失函数值
        Y_hat = np.zeros((m, self.K))                      # 定义预测值矩阵(m×K)

        for i in range(m):                                 # 计算每个数据的损失值
            Y_hat[i] = self.softmax(X[i])                  # 计算当前数据的预测值
            loss += np.dot(Y[i], np.log(Y_hat[i]))         # 将损失值累加到总的损失值

        return -1.0 * loss                                 # 返回损失函数值

    def stochastic_gradient_descent(self):  # 随机梯度下降法（不含偏置b）
        last_loss = self.compute_loss(self.X, self.Y)      # 计算损失函数值
        loss_list = [last_loss]                            # 保存损失函数值
        diff_loss = np.inf                                 # 设置损失值变化量为无穷大
        current_iter = 0                                   # 当前训练次数

        tol, max_iter = self.tol, self.max_iter            # 为缩短代码长度
        while diff_loss > tol or current_iter < max_iter:  # 继续训练的条件
            i = np.random.randint(self.m)                  # 从0到m-1随机选取一个整数
            Xi, Y, eta = self.X[i], self.Y, self.eta       # 为缩短代码长度

            y_hat = self.softmax(Xi)                       # 计算数据Xi的预测值
            for j in range(self.K):                        # 遍历各个类别的权重向量
                self.W[j] -= eta * (y_hat[j]-Y[i][j]) * Xi # 更新各个类别的权重向量

            loss = self.compute_loss(self.X, self.Y)       # 计算当前损失值
            diff_loss = np.abs(last_loss - loss)           # 计算损失函数值变化量
            loss_list.append(loss)                         # 将当前损失值加入损失值列表
            last_loss = loss                               # 将当前损失值设为上个损失值
            current_iter += 1                              # 训练次数加1

        return loss_list                                   # 返回损失值列表

    def fit(self):  # 拟合数据，训练模型
        self.stochastic_gradient_descent()                 # 使用随机梯度下降法训练

    def predict(self, X):  # 利用训练好的模型，计算预测值
        m = len(X)                                         # 获取数据长度
        X = np.c_[np.ones(m), X]                           # 预处理数据集

        Y_hat = np.zeros((m, self.K))                      # 定义预测值矩阵(m×K)
        for i in range(m):                                 # 遍历X的每个数据
            Y_hat[i] = self.softmax(X[i])                  # 计算每个数据的预测值

        y_hat = np.zeros(m)                                # 定义数据所属类别向量(m,)
        for i in range(m):                                 # 遍历X的每个数据
            y_hat[i] = np.argmax(Y_hat[i])                 # 获取每个数据所属类别

        return y_hat                                       # 返回数据所属类别向量(m,)

    def score(self, X, y):  # 计算训练或预测得分
        y_hat = self.predict(X)                            # 获取数据集X的预测值
        count = np.sum(y_hat == y)                         # 预测值与真实值相等的个数

        return count / len(y)                              # 返回训练或预测得分

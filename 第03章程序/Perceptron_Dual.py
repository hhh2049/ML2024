# encoding=utf-8
import numpy as np

class Perceptron_Dual:  # 感知机的实现（对偶形式）
    def __init__(self, X, y, eta=0.1, max_iter=1000):
        self.X         = X                               # 训练数据集(m,n)
        self.y         = y                               # 训练数据集的真实标签(m,)
        self.eta       = eta                             # 学习率
        self.max_iter  = max_iter                        # 最大训练次数
        self.m, self.n = X.shape                         # 数据集的数据量和特征数
        self.alpha     = np.zeros(self.m)                # 定义alpha向量
        self.Gram      = np.zeros((self.m, self.m))      # 定义Gram矩阵
        self.w         = np.zeros(self.n)                # 分离超平面的权重向量
        self.b         = 0.0                             # 分离超平面的偏置

    def compute_gram_matrix(self):  # 计算Gram矩阵
        for i in range(self.m):                          # 遍历所有训练数据
            for j in range(self.m):                      # Gram对称，可只计算一半
                X = self.X                               # 为缩短下一行代码长度
                self.Gram[i][j] = np.dot(X[i], X[j])     # 计算两个特征向量内积

    def compute_w_b(self):  # 基于alpha向量计算w、b
        w, b = 0.0, 0.0                                  # 定义w、b
        for i in range(self.m):                          # 依据公式计算w、b
            X, y = self.X, self.y                        # 为缩短以下两行代码长度
            w = w + self.alpha[i] * y[i] * X[i]          # 计算w
            b = b + self.alpha[i] * y[i]                 # 计算b

        return w, b                                      # 返回w、b

    def predict_one(self, i):  # 计算数据X[i]的预测值
        y_hat = 0.0                                      # 定义预测值
        for j in range(self.m):                          # 遍历所有数据
            alpha, y, G = self.alpha, self.y, self.Gram  # 为缩短下一行代码长度
            y_hat += alpha[j] * y[j] * G[j][i]           # 计算X[i]预测值
            y_hat += alpha[j] * y[j]                     # 计算X[i]预测值

        return y_hat                                     # 返回数据X[i]的预测值

    def fit(self):  # 拟合数据，训练模型
        self.compute_gram_matrix()                       # 计算Gram矩阵

        i = 0                                            # 训练次数计数
        while i < self.max_iter:                         # 未达到最大训练次数max_iter
            is_train_this_epoch = False                  # 本轮是否进行了训练
            index_random = np.arange(self.m)             # 生成训练数据集的下标0~m-1
            np.random.shuffle(index_random)              # 随机打乱数据集的下标

            for i in range(self.m):                      # 每轮训练遍历数据集
                i_ = index_random[i]                     # 获取一个随机的下标
                xi, yi = self.X[i_], self.y[i_]          # 获取一个数据及标签
                yi_hat = self.predict_one(i_)            # 计算数据X[i_]的预测值
                yi_hat = 1 if yi_hat >= 0 else -1        # 确定所属的分类

                if yi_hat != yi:                         # 分类错误，更新模型参数
                    alpha, eta = self.alpha, self.eta    # 为缩短下一行代码长度
                    self.alpha[i_] = alpha[i_] + eta     # 更新模型参数alpha
                    is_train_this_epoch = True           # 本轮执行了训练
                    break                                # 一轮只进行一次训练

            if is_train_this_epoch is False:             # 本轮未训练，分类全正确
                break                                    # 跳出循环，终止训练
            i = i + 1                                    # 训练次数加1

        self.w, self.b = self.compute_w_b()              # 训练终止，计算参数w、b

    def predict(self, X):  # 预测新数据所属的类
        y_predict = np.zeros(len(X))                    # 存放预测结果

        for i in range(len(X)):                         # 遍历X的每个数据
            yi_hat = np.dot(X[i], self.w) + self.b      # 计算预测值
            y_predict[i] = 1 if yi_hat >= 0 else -1     # 确定所属的分类

        return y_predict                                # 返回预测结果

    def score(self, X, y):  # 计算数据分类的正确率
        y_predict = self.predict(X)                     # 执行预测
        correct_count = np.sum(y_predict == y)          # 统计分类正确的数量

        return correct_count / len(X)                   # 返回正确率

def test():
    train_X = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # 训练数据集
    train_y = np.array([-1, -1, 1, -1])                   # 训练数据的真实标签

    pt = Perceptron_Dual(train_X, train_y, eta=0.5)       # 定义感知机类
    pt.fit()                                              # 拟合数据，训练模型

    print(pt.predict(train_X))                            # 打印训练结果
    print(pt.score(train_X, train_y))                     # 打印训练得分
    print(pt.w)                                           # 打印权重向量
    print(pt.b)                                           # 打印偏置

if __name__ == "__main__":
    test()

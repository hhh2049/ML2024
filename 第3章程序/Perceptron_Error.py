# encoding=utf-8
import numpy as np

class Perceptron_Error:  # 感知机的实现（误差形式）
    def __init__(self, X, y, eta=0.1, max_iter=1000):
        self.X         = X                            # 训练数据集(m,n)
        self.y         = y                            # 训练数据集的真实标签(m,)
        self.eta       = eta                          # 学习率
        self.max_iter  = max_iter                     # 最大训练次数
        self.m, self.n = X.shape                      # 数据集的数据量和特征数
        self.w         = np.zeros(self.n)             # 待学习的权重向量
        self.b         = 0.0                          # 待学习的偏置

    def fit(self):  # 拟合数据，训练模型
        i = 0                                         # 训练次数计数
        while i < self.max_iter:                      # 未达到最大训练次数max_iter
            is_train_this_epoch = False               # 标记本轮是否进行了训练
            index_random = np.arange(self.m)          # 生成训练数据集的下标0~m-1
            np.random.shuffle(index_random)           # 随机打乱数据集的下标

            for i in range(self.m):                   # 每轮训练遍历数据集
                xi = self.X[index_random[i]]          # 随机获取一个数据
                yi_hat = np.dot(xi, self.w) + self.b  # 计算预测值
                yi_hat = 1 if yi_hat >= 0 else 0      # 确定所属的分类
                yi = self.y[index_random[i]]          # 获取真实的分类

                if yi_hat != yi:                      # 分类错误，更新模型参数
                    eta = self.eta                    # 为缩短以下两行代码长度
                    self.w += eta * (yi-yi_hat) * xi  # 更新权重向量
                    self.b += eta * (yi-yi_hat)       # 更新偏置
                    is_train_this_epoch = True        # 标记本轮执行了训练
                    break                             # 一轮只进行一次训练

            if is_train_this_epoch is False:          # 本轮未训练，分类全正确
                break                                 # 跳出循环，终止训练
            i = i + 1                                 # 训练次数加1

    def predict(self, X):  # 预测新数据所属的类
        y_predict = np.zeros(len(X))                  # 存放预测结果

        for i in range(len(X)):                       # 遍历X的每个数据
            yi_hat = np.dot(X[i], self.w) + self.b    # 计算预测值
            y_predict[i] = 1 if yi_hat >= 0 else 0    # 确定所属的分类

        return y_predict                              # 返回预测结果

    def predict2(self, X):  # 预测新数据所属的类（简化形式）
        y_hat = np.dot(X, self.w) + self.b            # 计算预测值
        y_predict = np.where(y_hat >= 0, 1, 0)        # 获取分类值

        return y_predict                              # 返回预测结果

    def score(self, X, y):  # 计算数据分类的正确率
        y_predict = self.predict(X)                   # 执行预测
        correct_count = np.sum(y_predict == y)        # 统计分类正确的数量

        return correct_count / len(X)                 # 返回正确率


def test():
    train_X = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # 训练数据集
    train_y = np.array([0, 0, 1, 0])                      # 训练数据的真实标签

    pt = Perceptron_Error(train_X, train_y, eta=0.5)      # 定义感知机类
    pt.fit()                                              # 拟合数据，训练模型

    print(pt.predict(train_X))                            # 打印训练结果
    print(pt.score(train_X, train_y))                     # 打印训练得分
    print(pt.w)                                           # 打印权重向量
    print(pt.b)                                           # 打印偏置

if __name__ == "__main__":
    test()

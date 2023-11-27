# encoding=utf-8
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

class Perceptron_Pocket:  # 口袋感知机的实现（原始形式）
    def __init__(self, X, y, eta=0.1, max_iter=1000):
        self.X         = X                             # 训练数据集(m,n)
        self.y         = y                             # 训练数据的真实标签(m,)
        self.eta       = eta                           # 学习率
        self.max_iter  = max_iter                      # 最大训练次数
        self.m, self.n = X.shape                       # 数据集的数据量和特征数
        self.w         = np.zeros(self.n)              # 待学习的权重向量
        self.b         = 0.0                           # 待学习的偏置
        self.score_max = 0.0                           # 训练时最高得分
        self.w_max     = np.zeros(self.n)              # 训练得分最高时的w
        self.b_max     = 0.0                           # 训练得分最高时的b

    def fit(self):  # 拟合数据，训练模型
        for _ in range(self.max_iter):                 # 最多执行max_iter次训练
            is_train_this_epoch = False                # 本轮是否进行了训练
            index_random = np.arange(self.m)           # 生成训练数据集的下标0~m-1
            np.random.shuffle(index_random)            # 随机打乱数据集的下标

            for i in range(self.m):                    # 每轮训练遍历数据集
                xi = self.X[index_random[i]]           # 随机获取一个数据
                yi_hat = np.dot(xi, self.w) + self.b   # 计算预测值
                yi_hat = 1 if yi_hat >= 0 else -1      # 确定所属的分类
                yi = self.y[index_random[i]]           # 获取正确的分类值

                if yi_hat != yi:                       # 分类错误，更新模型参数
                    self.w += self.eta * yi * xi       # 更新权重向量
                    self.b += self.eta * yi            # 更新偏置
                    is_train_this_epoch = True         # 本轮执行了训练
                    self.update_max_w_b()              # 更新相关值
                    break                              # 一轮只进行一次训练

            if is_train_this_epoch is False:           # 本轮未训练，分类全正确
                break                                  # 跳出循环，终止训练

    def predict(self, X):  # 预测新数据所属的类
        y_predict = np.zeros(len(X))                   # 存放预测结果

        for i in range(len(X)):                        # 遍历X的每个数据
            yi_hat = np.dot(X[i], self.w) + self.b     # 计算预测值
            y_predict[i] = 1 if yi_hat >= 0 else -1    # 确定所属的分类

        return y_predict                               # 返回预测结果

    def score(self, X, y):  # 计算数据分类的正确率
        y_predict = self.predict(X)                    # 执行预测
        correct_count = np.sum(y_predict == y)         # 统计分类正确的数量

        return correct_count / len(X)                  # 返回正确率

    def update_max_w_b(self):  # 更新相关值
        score = self.score(self.X, self.y)             # 计算当前训练得分
        if score > self.score_max:                     # 如果当前得分更高
            self.score_max = score                     # 保存当前得分
            self.w_max = self.w                        # 保存当前w
            self.b_max = self.b                        # 保存当前b

def main():
    iris = load_iris()                                      # 载入鸢尾花数据集
    X, y = iris.data[50:150, :], iris.target[50:150]        # 截取后100条数据
    y[:50] = 1                                              # 前50条数据赋值1
    y[50:100] = -1                                          # 后50条数据赋值-1

    skl_pt = Perceptron()                                   # 定义官方库感知机
    skl_pt.fit(X, y)                                        # 拟合数据，训练模型
    skl_score = skl_pt.score(X, y)                          # 计算训练得分
    print("sklearn Perceptron score    = %f" % skl_score)   # 打印训练得分

    our_pt = Perceptron_Pocket(X, y)                        # 定义自实现口袋感知机
    our_pt.fit()                                            # 拟合数据，训练模型
    our_score = our_pt.score(X, y)                          # 计算训练得分
    print("our Perceptron_Pocket score = %f" % our_score)   # 打印训练得分

if __name__ == "__main__":
    main()

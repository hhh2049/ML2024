# encoding=utf-8
import numpy as np
from Perceptron_Primal import Perceptron_Primal

def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])     # 原始训练数据集
    train_y = np.array([-1, 1, 1, -1])                 # 训练数据的真实标签

    x1, x2 = X[:, 0], X[:, 1]                          # 获取X的第1列和第2列
    train_X = np.c_[x1, x2, x1**2, x1*x2, x2**2]       # 构造多项式特征

    pt = Perceptron_Primal(train_X, train_y, eta=0.5)  # 定义感知机类对象
    pt.fit()                                           # 拟合数据，训练模型
    print("score = %f" % pt.score(train_X, train_y))   # 打印训练得分
    print(pt.w, pt.b)                                  # 打印权重向量和偏置

if __name__ == "__main__":
    main()

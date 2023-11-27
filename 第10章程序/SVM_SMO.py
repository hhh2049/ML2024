# encoding=utf-8
import numpy as np

class SVM_SMO:
    def __init__(self, X, y, C=1.0, tol=0.001, kernel='linear', gamma=0.01):  # 初始化函数
        self.X   = X                                               # 训练数据集
        self.y   = y                                               # 分类标签（1或-1）
        self.C   = C                                               # 惩罚参数C
        self.tol = tol                                             # 精度阈值ε，硬间隔为0

        if kernel == 'linear':                                     # 如果是线性核
            self.K = self.linear_kernel                            # K为线性核函数
        elif kernel == "rbf":                                      # 如果是高斯核
            self.K = self.gaussian_kernel                          # K为高斯核函数
        self.gamma = gamma                                         # 高斯核的γ参数

        self.m, self.n = self.X.shape                              # 数据量和特征数
        self.a = np.zeros(self.m)                                  # 待求的alpha向量
        self.b = 0.0                                               # 待求的偏置b

    def linear_kernel(self, x1, x2):  # 线性核函数
        return np.dot(x1, x2)                                      # 计算线性核函数值

    def gaussian_kernel(self, x1, x2):  # 高斯核函数
        p = np.dot(x1 - x2, x1 - x2)                               # 两个向量差的内积
        return np.exp(-self.gamma * p)                             # 计算高斯核函数值

    def g(self, xi):  # 计算一个数据的预测值，基于式(10.44)
        g_xi = 0.0                                                 # 用于存储预测值
        for j in range(self.m):                                    # 遍历所有数据
            g_xi += self.a[j] * self.y[j] * self.K(self.X[j], xi)  # 计算预测值
        g_xi += self.b                                             # 加上偏置项

        return g_xi                                                # 返回预测值

    def get_L_H(self, i, j):  # 计算L和H，基于式(10.50)和式(10.51)
        if self.y[i] != self.y[j]:                                 # 如果y1≠y2
            L = max(0, self.a[j] - self.a[i])                      # 根据公式计算L、H
            H = min(self.C, self.C + self.a[j] - self.a[i])
        else:                                                      # 如果y1=y2
            L = max(0, self.a[j] + self.a[i] - self.C)             # 根据公式计算L、H
            H = min(self.C, self.a[j] + self.a[i])

        return L, H                                                # 返回L、H

    def compute_a1new_a2new(self, i, j, Ei, Ej):  # 计算a1_new和a2_new
        X, y = self.X, self.y                                      # 为缩短代码长度

        eta = self.K(X[i], X[i]) + self.K(X[j], X[j])              # 计算Kii+Kjj-2Kij
        eta -= 2 * self.K(X[i], X[j])                              # 计算Kii+Kjj-2Kij
        if eta <= 0:                                               # 如果eta小于等于0
            return self.a[i], self.a[j], 0                         # 返回a1、a2、0
        a2_new_unc = self.a[j] + y[j] * (Ei - Ej) / eta            # 计算未经剪辑的a2

        L, H = self.get_L_H(i, j)                                  # 计算L和H
        if L == H:                                                 # 如果相等，无调整空间
            return self.a[i], self.a[j], 0                         # 返回a1、a2、0
        if a2_new_unc > H:   a2_new = H                            # 计算a2_new
        elif a2_new_unc < L: a2_new = L
        else:                a2_new = a2_new_unc

        threshold = self.tol * (a2_new + self.a[j] + self.tol)     # 计算调整阈值
        if np.abs(a2_new - self.a[j]) < threshold:                 # a2调整量过小
            return self.a[i], self.a[j], 0                         # 返回a1、a2、0

        a1_new = self.a[i] + (self.a[j] - a2_new) * y[i] * y[j]    # 计算a1_new

        return a1_new, a2_new, 1                                   # 返回更新后的α1和a2

    def compute_b_new(self, i, j, a1_new, a2_new, Ei, Ej):  # 计算b_new
        a, b, C = self.a, self.b, self.C                           # 为缩短代码长度
        X, y, K = self.X, self.y, self.K

        b1_new = -Ei + (a[i] - a1_new) * y[i] * K(X[i], X[i]) + \
                 (a[j] - a2_new) * y[j] * K(X[j], X[i]) + b        # 计算b1_new
        if 0 < a1_new < C: return b1_new

        b2_new = -Ej + (a[i] - a1_new) * y[i] * K(X[i], X[j]) + \
                 (a[j] - a2_new) * y[j] * K(X[j], X[j]) + b        # 计算b2_new
        if 0 < a2_new < C: return b2_new

        return (b1_new + b2_new) / 2                               # 返回两者均值

    def is_not_fit_KKT(self, i):  # 依据式(10.57)，判断一个数据是否违反KKT条件（方法一）
        ui = self.y[i] * self.g(self.X[i])
        if self.a[i] < self.C and ui + self.tol < 1:
            return True
        if self.a[i] > 0 and ui - self.tol > 1:
            return True

        return False

    def train(self):  # 拟合数据，训练模型
        alpha_changed_total = 0  # 用于统计alpha更新次数（实际训练次数）
        while True:
            alpha_changed = 0  # 本轮训练alpha更新次数（循环一次alpha的更新次数）

            for i in range(self.m):  # 首先优化：在间隔边界上且违反KKT条件的数据Xi
                if 0 < self.a[i] < self.C and self.is_not_fit_KKT(i):
                    alpha_changed += self.optimize_a1(i)

            if alpha_changed == 0:
                for i in range(self.m):  # 其次优化：遍历数据集寻找违反KKT条件的数据Xi
                    if self.is_not_fit_KKT(i):
                        alpha_changed += self.optimize_a1(i)

            if alpha_changed == 0:  # 所有α均符合KKT条件，跳出循环，训练结束
                break

            alpha_changed_total += alpha_changed  # 统计alpha的更新次数
            print("alpha changed total = ", alpha_changed_total)  # 打印alpha总的更新次数
            if alpha_changed_total >= 2 * self.m:  # 超过阈值，停止训练，防止长时间不收敛
                print("I am here")
                break

    def optimize_a1(self, i):  # 寻找a2并优化α1、α2
        mask = (self.a > 0) & (self.a < self.C)                    # 筛选条件0<α<C
        bound_index     = np.where(mask)[0]                        # 筛选0<α<C
        non_bound_index = np.where(~mask)[0]                       # 筛选α=0或α=C
        Ei = self.g(self.X[i]) - self.y[i]                         # 计算E1

        for j in bound_index:   # 在边界数据中，随机选择a2优化       # 遍历边界下标
            Ej = self.g(self.X[j]) - self.y[j]                     # 计算E2
            if self.optimize_a1_a2(i, j, Ei, Ej):                  # 如果本次优化成功
                return 1                                           # 返回优化成功次数

        for j in non_bound_index:  # 在非边界数据中，随机选择a2优化  # 遍历非边界下标
            Ej = self.g(self.X[j]) - self.y[j]                     # 计算E2
            if self.optimize_a1_a2(i, j, Ei, Ej):                  # 如果本次优化成功
                return 1                                           # 返回优化成功次数

        return 0                                                   # 返回优化失败标志

    def optimize_a1_a2(self, i, j, Ei, Ej):  # 优化a1和a2
        if i == j: return 0                                        # 若α1=α2则返回

        a1_new,a2_new,ret = self.compute_a1new_a2new(i, j, Ei, Ej) # 优化α1、α2
        if ret == 0: return 0                                      # 优化失败提前返回
        b_new = self.compute_b_new(i, j, a1_new, a2_new, Ei, Ej)   # 计算新的b
        self.a[i], self.a[j], self.b = a1_new, a2_new, b_new       # 更新α1、α2、b

        return 1                                                   # 返回优化成功标志

    def score(self, X, y):  # 计算预测的正确率
        count = 0                                                  # 计数预测正确数量
        for i in range(len(X)):                                    # 遍历当前的数据集
            y_hat = self.g(X[i])                                   # 计算预测值
            y_hat = 1 if y_hat >= 0 else -1                        # 确定分类标签
            if y_hat == y[i]: count += 1                           # 预测正确计数加一

        return count / len(X)                                      # 返回预测得分

    # 以下函数用于测试，不参与实际训练和预测
    def is_not_fit_KKT2(self, i):  # 判断一个数据是否违反KKT条件（方法二）
        a, C, g   = self.a, self.C, self.g
        X, Y, tol = self.X, self.y, self.tol

        if a[i] == 0 and Y[i] * g(X[i]) + tol < 1:
            return True
        if 0 < a[i] < C and np.abs(Y[i] * g(X[i]) - 1) > tol:
            return True
        if a[i] == C and Y[i] * g(X[i]) - tol > 1:
            return True
        return False

    def is_not_fit_KKT3(self, i):  # 判断一个数据是否违反KKT条件（方法三）
        if -self.tol <= 1-self.y[i]*self.g(self.X[i]) <= self.tol:
            return False
        else:
            return True

    def compute_loss(self):  # 根据式(10.35)，计算损失函数值
        a, X, Y, K = self.a, self.X, self.y, self.K                # 为缩短代码长度

        loss = 0.0                                                 # 存放损失函数值
        for i in range(self.m):                                    # 计算损失函数值
            for j in range(self.m):
                loss += 0.5*a[i]*a[j]*Y[i]*Y[j]*K(X[i],X[j])
            loss = loss - a[i]

        return loss                                                # 返回损失函数值

    def compute_w(self):  # 根据式(10.27)，计算线性核的权重向量
        w = np.zeros(self.n)                                       # 存放权重向量
        if self.K != self.linear_kernel: return w                  # 若非线性核则返回
        for i in range(self.m):                                    # 遍历所有训练数据
            w = w + self.a[i] * self.y[i] * self.X[i]              # 计算权重向量

        return w                                                   # 返回权重向量

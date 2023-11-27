# encoding=utf-8
import numpy as np

class HMM:  # 隐马尔可夫模型，实现Baum-Welch学习算法
    def __init__(self, n_states, n_observations, observations, n_iter=10, tol=1e-5):
        self.pi     = np.zeros(n_states)                         # 初始状态概率向量
        self.A      = np.zeros((n_states, n_states))             # 状态转移概率矩阵
        self.B      = np.zeros((n_states, n_observations))       # 观测生成概率矩阵
        self.o_list = observations                               # 给定的观测序列
        self.n      = n_states                                   # 状态数
        self.m      = n_observations                             # 观测数
        self.T      = len(observations)                          # 时间数
        self.n_iter = n_iter                                     # 最大训练次数
        self.tol    = tol                                        # 训练收敛阈值

    def initialize_hmm_model(self):  # 随机初始化模型参数
        self.pi = np.random.rand(self.n)                         # 生成n个0~1均匀分布样本
        self.pi = self.pi / self.pi.sum()                        # 归一化，概率之和为1

        for i in range(self.n):                                  # 遍历矩阵A的每一行
            self.A[i] = np.random.rand(self.n)                   # 生成n个0~1均匀分布样本
            self.A[i] = self.A[i] / self.A[i].sum()              # 归一化:A每行概率之和为1

        for i in range(self.n):                                  # 遍历矩阵B的每一行
            self.B[i] = np.random.rand(self.m)                   # 生成m个0~1均匀分布样本
            self.B[i] = self.B[i] / self.B[i].sum()              # 归一化:B每行概率之和为1

    def forward(self):  # 计算前向概率
        alpha = np.zeros((self.T, self.n))                       # 用于存储alpha值

        o1 = self.o_list[0]                                      # 获取第一个观测
        for i in range(self.n):                                  # 初始化alpha的第一行
            alpha[0][i] = self.pi[i] * self.B[i][o1]             # 计算alpha第一行的值

        A, B = self.A, self.B                                    # 为缩短代码长度
        for t in range(self.T-1):                                # 根据公式计算前向概率
            for i in range(self.n):                              # 对每个t，计算αt+1(i)
                sum_p = 0.0                                      # 用于存放αt+1(i)
                ot1 = self.o_list[t+1]                           # 获取ot+1
                for j in range(self.n):                          # 算法复杂度O(Tn^2)
                    sum_p += alpha[t][j] * A[j][i] * B[i][ot1]   # 计算αt+1(i)
                alpha[t+1][i] = sum_p                            # 保存αt+1(i)

        return alpha                                             # 返回alpha

    def backward(self):  # 计算后向概率
        beta = np.zeros((self.T, self.n))                        # 用于存储beta值

        A, B = self.A, self.B                                    # 为缩短代码长度
        beta[self.T-1] = np.ones(self.n)                         # 初始化beta最后一行为1
        for t in range(self.T-2, -1, -1):                        # 根据公式计算后向概率
            for i in range(self.n):                              # 对每个t，计算βt(i)
                sum_p = 0.0                                      # 存放βt(i)
                ot1   = self.o_list[t+1]                         # 获取ot+1
                for j in range(self.n):                          # 算法复杂度O(Tn^2)
                    sum_p += beta[t+1][j] * A[i][j] * B[j][ot1]  # 计算βt(i)
                beta[t][i] = sum_p                               # 保存βt(i)

        return beta                                              # 返回beta

    def log_likelihood(self):  # 计算对数似然函数值
        alpha = self.forward()                                   # 计算前向概率

        sum_p = 0.0                                              # 用于存储P(O)
        for i in range(self.n):                                  # 遍历alpha最后一行
            sum_p += alpha[self.T-1][i]                          # 计算P(O)的值

        return np.log(sum_p)                                     # 计算并返回logP(O)

    def _E_step(self):  # EM算法的E步，基于当前模型参数λ=(A、B、π)计算ξ(xi)和γ(gamma)
        alpha, beta = self.forward(), self.backward()            # 计算前向和后向概率

        xi = np.zeros((self.T, self.n, self.n))                  # 用于存储ξ(xi)张量
        for t in range(self.T - 1):                              # 遍历时间T
            for i in range(self.n):                              # 遍历状态n
                for j in range(self.n):                          # 遍历状态n
                    ot1 = self.o_list[t + 1]                     # 获取ot+1
                    a, b = alpha, beta                           # 为缩短代码长度
                    A_ij, B_jot1 = self.A[i][j], self.B[j][ot1]  # 为缩短代码长度
                    p_ij_o = a[t][i] * A_ij * B_jot1 * b[t+1][j] # 计算ξt(i,j)的分子
                    p_o = 0.0                                    # 用于存储P(O)
                    for k in range(self.n):                      # 遍历状态n
                        p_o += alpha[t][k] * beta[t][k]          # 计算P(O)的值
                    xi[t][i][j] = p_ij_o / p_o                   # 计算ξt(i,j)

        gamma = np.zeros((self.T, self.n))                       # 用于存储γ(gamma)矩阵
        for t in range(self.T):                                  # 遍历时间T
            for i in range(self.n):                              # 遍历状态n
                p_i_o = alpha[t][i] * beta[t][i]                 # 用于存储P(i,O)
                p_o = 0.0                                        # 用于存储P(O)
                for j in range(self.n):                          # 遍历状态n
                    p_o += alpha[t][j] * beta[t][j]              # 计算P(O)的值
                gamma[t][i] = p_i_o / p_o                        # 计算γt(i)

        return xi, gamma                                         # 返回ξ(xi)和γ(gamma)

    def _M_step(self, xi, gamma):  # EM算法的M步，更新模型参数λ=(A、B、π)
        for i in range(self.n):                                  # 计算并更新状态转移矩阵A
            for j in range(self.n):                              # 遍历状态n
                a, b = 0.0, 0.0                                  # ab分别存储分子分母的值
                for t in range(self.T - 1):                      # 遍历时间T
                    a += xi[t][i][j]                             # 计算分子a的值
                    b += gamma[t][i]                             # 计算分母b的值
                self.A[i][j] = a / b                             # 计算并更新Aij

        for j in range(self.n):                                  # 计算并更新观测概率矩阵B
            for k in range(self.m):                              # 遍历观测m
                a, b = 0.0, 0.0                                  # ab分别存储分子分母的值
                for t in range(self.T):                          # 遍历时间T
                    if self.o_list[t] == k:                      # 如果ot==vk
                        a += gamma[t][j]                         # 计算分子a的值
                    b += gamma[t][j]                             # 计算分母b的值
                self.B[j][k] = a / b                             # 计算并更新Bjk

        for i in range(self.n):                                  # 计算更新初始状态概率π
            self.pi[i] = gamma[0][i]                             # 遍历所有状态，更新π值

    # EM算法的E步（简化版），基于当前模型参数λ=(A、B、π)计算前向和后向概率
    def _E_step_simple(self):
        alpha = self.forward()                                   # 计算前向概率
        beta  = self.backward()                                  # 计算后向概率

        return alpha, beta                                       # 返回前向概率和后向概率

    # EM算法的M步（简化版），更新模型参数λ=(A、B、π)
    def _M_step_simple(self, alpha, beta):
        for i in range(self.n):                                  # 计算并更新状态转移矩阵A
            for j in range(self.n):                              # 遍历状态n
                a, b = 0.0, 0.0                                  # ab分别存储分子分母的值
                for t in range(self.T - 1):                      # 遍历时间T
                    ot1 = self.o_list[t+1]                       # 获取ot+1
                    al, be = alpha, beta                         # 为缩短代码长度
                    A_ij, B_jot1 = self.A[i][j], self.B[j][ot1]  # 为缩短代码长度
                    a += al[t][i] * A_ij * B_jot1 * be[t+1][j]   # 计算分子a的值
                    b += alpha[t][i] * beta[t][i]                # 计算分母b的值
                self.A[i][j] = a / b                             # 计算并更新Aij

        for j in range(self.n):                                  # 计算并更新观测概率矩阵B
            for k in range(self.m):                              # 遍历观测m
                a, b = 0.0, 0.0                                  # ab分别存储分子分母的值
                for t in range(self.T):                          # 遍历时间T
                    if self.o_list[t] == k:                      # 如果ot==vk
                        a += alpha[t][j] * beta[t][j]            # 计算分子a的值
                    b += alpha[t][j] * beta[t][j]                # 计算分母b的值
                self.B[j][k] = a / b                             # 计算并更新Bjk

        for i in range(self.n):                                  # 计算并更新初始状态概率π
            a = alpha[0][i] * beta[0][i]                         # 计算分子a的值
            b = 0.0                                              # 存储分母b的值
            for j in range(self.n):                              # 遍历状态n
                b += alpha[0][j] * beta[0][j]                    # 计算分母b的值
            self.pi[i] = a / b                                   # 计算并更新πi

    def train(self):  # 执行训练
        np.set_printoptions(precision=2)                         # 设置打印精确度
        self.initialize_hmm_model()                              # 初始化模型参数

        train_count = 0                                          # 训练次数计数
        delta = np.inf                                           # 对数似然值的变化量
        last_likelihood = self.log_likelihood()                  # 计算初始对数似然值

        while train_count < self.n_iter and delta > self.tol:    # 对数似然变化量大于阈值
            # 普通版
            # xi, gamma = self._E_step()                         # 执行EM算法的E步
            # self._M_step(xi, gamma)                            # 执行EM算法的M步

            # 精简版
            alpha, beta = self._E_step_simple()                  # 执行EM算法的E步
            self._M_step_simple(alpha, beta)                     # 执行EM算法的M步

            # 重新计算对数似然函数值
            likelihood = self.log_likelihood()                   # 重新计算对数似然值
            delta = likelihood - last_likelihood                 # 计算对数似然变化量
            last_likelihood = likelihood                         # 更新对数似然函数值
            train_count += 1                                     # 更新训练次数

            print("count      = %d" % train_count)               # 打印当前训练次数
            print("likelihood = %f" % likelihood)                # 打印当前对数似然值
            print("delta      = %f\n" % delta)                   # 打印对数似然变化量

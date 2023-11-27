# encoding=utf-8
import numpy as np

class HMM:  # 隐马尔可夫模型：实现采样算法、前向算法、后向算法、预测（解码）算法（维特比算法）
    def __init__(self, pi, A, B, o_list, states=None, observations=None):
        self.pi      = pi                                        # 初始状态概率向量
        self.A       = A                                         # 状态转移概率矩阵
        self.B       = B                                         # 观测生成概率矩阵
        self.o_list  = o_list                                    # 给定的观测序列
        self.I_names = states                                    # 状态序列的名称(可为空)
        self.O_names = observations                              # 观测序列的名称(可为空)
        self.n, m    = B.shape                                   # 状态数和观测数
        self.T       = len(o_list)                               # 观测序列的长度

    def simulate(self):  # 观测序列的生成（采样）算法
        pi, A, B, T = self.pi, self.A, self.B, self.T            # 为缩短代码长度
        states = np.zeros(T, dtype=int)                          # 用于存放状态序列
        observ = np.zeros(T, dtype=int)                          # 用于存放观测序列

        s0 = np.random.multinomial(1, pi).argmax()               # 根据向量π生成第1个状态
        states[0] = s0                                           # 保存第一个状态
        observ[0] = np.random.multinomial(1, B[s0]).argmax()     # 根据矩阵B生成第1个观测

        for t in range(1, T):                                    # 循环T-1次
            st_1 = states[t-1]                                   # 获取上一个状态
            st = np.random.multinomial(1, A[st_1]).argmax()      # 根据矩阵A生成下一状态
            states[t] = st                                       # 保存当前的状态
            observ[t] = np.random.multinomial(1, B[st]).argmax() # 根据矩阵B生成当前观测

        return states, observ                                    # 返回状态序列观测序列

    def forward(self):  # 计算前向概率
        alpha = np.zeros((self.T, self.n))                       # 用于存储alpha值

        o1 = self.o_list[0]                                      # 获取第一个观测
        for i in range(self.n):                                  # 初始化alpha的第一行
            alpha[0][i] = self.pi[i] * self.B[i][o1]             # 计算alpha第一行的值

        A, B = self.A, self.B                                    # 为缩短代码长度
        for t in range(self.T-1):                                # 根据公式计算前向概率
            for i in range(self.n):                              # 对每个t，计算αt+1(i)
                sum_p = 0.0                                      # 用于存放αt+1(i)
                ot1   = self.o_list[t+1]                         # 获取ot+1
                for j in range(self.n):                          # 算法复杂度O(Tn^2)
                    sum_p += alpha[t][j] * A[j][i] * B[i][ot1]   # 计算αt+1(i)
                alpha[t+1][i] = sum_p                            # 保存αt+1(i)

        PO = 0.0                                                 # 用于存放P(O|λ)
        for i in range(self.n):                                  # 对alpha最后一行求和
            PO += alpha[self.T - 1][i]                           # 计算P(O|λ)
        return PO                                                # 返回P(O|λ)

    def forward2(self):  # 计算前向概率，使用numpy库函数
        alpha = np.zeros((self.T, self.n))                       # 用于存储alpha值

        o1 = self.o_list[0]                                      # 获取第一个观测
        alpha[0, :] = self.pi * self.B[:, o1]                    # 计算α1(i)

        a, A, B = alpha, self.A, self.B                          # 为缩短代码长度
        for t in range(self.T-1):                                # 根据公式计算前向概率
            for i in range(self.n):                              # 对每个t，计算αt+1(i)
                ot1 = self.o_list[t+1]                           # 获取ot+1
                a[t+1, i] = np.sum(a[t, :] * A[:, i]) * B[i,ot1] # 计算αt+1(i)

        return np.sum(alpha[-1])                                 # 返回P(O|λ)

    def backward(self):  # 计算后向概率
        beta = np.zeros((self.T, self.n))                        # 用于存储beta值

        A, B = self.A, self.B                                    # 为缩短代码长度
        beta[self.T-1] = np.ones(self.n)                         # beta最后一行初始化为1
        for t in range(self.T-2, -1, -1):                        # t=T-2,T-3,…,0
            for i in range(self.n):                              # 对每个t，计算βt(i)
                sum_p = 0.0                                      # 存放βt(i)
                ot1   = self.o_list[t+1]                         # 获取ot+1
                for j in range(self.n):                          # 算法复杂度O(Tn^2)
                    sum_p += A[i][j] * B[j][ot1] * beta[t+1][j]  # 计算βt(i)
                beta[t][i] = sum_p                               # 保存βt(i)

        PO = 0.0                                                 # 存放后向概率
        o1 = self.o_list[0]                                      # 获取o1
        for i in range(self.n):                                  # 循环n次
            PO += self.pi[i] * self.B[i][o1] * beta[0][i]        # 计算后向概率
        return PO                                                # 返回后向概率

    def decode(self):  # 预测（解码）算法（维特比算法）
        delta = np.zeros((self.T, self.n))                       # 用于存储δ的值
        psi   = np.zeros((self.T, self.n))                       # 用于存储ψ的值
        for i in range(self.n):                                  # 初始化
            o1 = self.o_list[0]                                  # 获取第一个观测
            delta[0][i] = self.pi[i] * self.B[i][o1]             # 计算δ的第一行

        A, B = self.A, self.B                                    # 为缩短代码长度
        for t in range(1, self.T):                               # t=1,2,…,T-1
            for j in range(self.n):                              # 遍历所有状态
                max_p = 0.0                                      # 存储最大概率
                max_i = -1                                       # 存储最大概率的上一状态
                for i in range(self.n):                          # 算法复杂度O(Tn^2)
                    ot = self.o_list[t]                          # 获取当前观测
                    p = delta[t-1][i] * A[i][j] * B[j][ot]       # 获取当前概率
                    if p > max_p:                                # 如果当前概率更大
                        max_p = p                                # 将当前概率作为最大概率
                        max_i = i                                # 更新相应的状态
                delta[t][j] = max_p                              # 保存最大概率
                psi[t][j]   = max_i                              # 保存相应的状态

        p_max = np.max(delta[self.T - 1])                        # 获取δ最后一行的最大值
        i_max_list = np.zeros(self.T, dtype=int)                 # 存储概率最大的状态序列
        i_max_list[self.T-1] = np.argmax(delta[self.T-1])        # 获取并保存最后一个状态
        for t in range(self.T-2, -1, -1):                        # 回溯，t=T-2,T-3,…,0
            last_i        = i_max_list[t+1]                      # 获取上一个状态
            i_max_list[t] = psi[t+1][last_i]                     # 回溯，获取状态序列

        return p_max, i_max_list                                 # 返回最大概率和状态序列

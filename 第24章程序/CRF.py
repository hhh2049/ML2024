# encoding=utf-8
import numpy as np

class CRF:  # 条件随机场预测算法(维特比算法)实现
    def __init__(self, tr, w1, st, w2, T, n, x_list, y_list):
        self.tr     = tr                                  # 转移特征函数集合
        self.w1     = w1                                  # 转移特征函数权重
        self.st     = st                                  # 状态特征函数集合
        self.w2     = w2                                  # 状态特征函数权重
        self.T      = len(x_list) - 1                     # 序列长度
        self.n      = len(y_list) - 1                     # 标签数量
        self.x_list = x_list                              # 输入序列
        self.y_list = y_list                              # 标签集合

    def w_Ft(self, j, i, xt):  # 计算w * Fi（yt-1, yt, xt）的值
        tr, st    = self.tr, self.st                      # 为缩短代码长度
        y, w1, w2 = self.y_list, self.w1, self.w2         # 为缩短代码长度
        result    = 0.0                                   # 存储计算结果

        for k in range(1, len(self.tr)):                  # 遍历转移特征函数
            result += w1[k] * tr[k](y[j], y[i], xt)       # 计算转移特征之和

        for l in range(1, len(self.st)):                  # 遍历状态特征函数
            result += w2[l] * st[l](y[i], xt)             # 计算状态特征之和

        return result                                     # 返回计算结果

    def viterbi(self): # 实现维特比算法
        w_Ft, x, y = self.w_Ft, self.x_list, self.y_list  # 为缩短代码长度
        delta = np.zeros((self.T + 1, self.n + 1))        # 定义δ矩阵，下标从1开始
        psi   = np.zeros((self.T + 1, self.n + 1))        # 定义ψ矩阵，下标从1开始

        start = 0                                         # start取值为0，表示初始化
        for i in range(1, self.n + 1):                    # 计算δ矩阵和ψ矩阵的第一行
            delta[1][i] = w_Ft(start, i, x[1])            # 计算δ矩阵的第一行
            psi[1][i]   = start                           # 计算ψ矩阵的第一行（无用）

        for t in range(2, self.T + 1):                    # 递推，t=2,3,...,T
            for i in range(1, self.n + 1):                # 计算δt(i)，i=1,2,...,n
                max_p, max_j = -1, -1                     # 存放概率最大值和下标
                for j in range(1, self.n + 1):            # 遍历所有j，j=1,2,...,n
                    p = delta[t-1][j] + w_Ft(j, i, x[t])  # 计算未归一化条件概率
                    if p > max_p:                         # 如果当前的概率更大
                        max_p, max_j = p, j               # 更新概率最大值和下标
                delta[t][i], psi[t][i] = max_p, max_j     # 存储概率最大值和下标

        print(delta[1:, 1:])                              # 打印δ矩阵，行列下标均从1开始
        print(psi[1:, 1:])                                # 打印ψ矩阵，行列下标均从1开始

        max_p = np.max(delta[self.T])                     # 获取未归一化条件概率的最大值
        max_y_list = np.zeros(self.T + 1, dtype="int")    # 定义概率最大的标注序列
        max_y_list[self.T] = np.argmax(delta[self.T])     # 获取概率最大值下标
        for j in range(self.T-1, 0, -1):                  # 回溯，获取概率最大的标注序列
            index = int(max_y_list[j+1])                  # 获取y_t+1的下标
            max_y_list[j] = psi[j+1][index]               # 获取y_t的下标

        return max_p, max_y_list                          # 返回概率最大值和最优标签序列

def main():
    # 定义转移特征函数
    def tr1(yt_1, yt, xt):
        return 1 if yt_1 == "p" and yt == "v" and xt == "love" else 0

    def tr2(yt_1, yt, xt):
        return 1 if yt_1 == "v" and yt == "v" and xt == "love" else 0

    def tr3(yt_1, yt, xt):
        return 1 if yt_1 == "p" and yt == "p" and xt == "you" else 0

    def tr4(yt_1, yt, xt):
        return 1 if yt_1 == "v" and yt == "p" and xt == "you" else 0

    # 定义状态特征函数
    def st1(yt, xt):
        return 1 if yt == "p" and xt == "I" else 0

    def st2(yt, xt):
        return 1 if yt == "v" and xt == "I" else 0

    def st3(yt, xt):
        return 1 if yt == "p" and xt == "love" else 0

    def st4(yt, xt):
        return 1 if yt == "v" and xt == "love" else 0

    def st5(yt, xt):
        return 1 if yt == "p" and xt == "you" else 0

    def st6(yt, xt):
        return 1 if yt == "v" and xt == "you" else 0

    tr = [None, tr1, tr2, tr3, tr4]                       # 定义转移特征函数集合
    w1 = [None, 0.95, 0.05, 0.15, 0.85]                   # 定义转移特征函数权重
    st = [None, st1, st2, st3, st4, st5, st6]             # 定义状态特征函数集合
    w2 = [None, 0.9, 0.1, 0.3, 0.7, 0.8, 0.2]             # 定义状态特征函数权重

    x_list = [None, "I", "love", "you"]                   # 定义输入序列
    y_list = [None, "p", "v"]                             # 定义标签集合

    crf = CRF(tr, w1, st, w2, 3, 2, x_list, y_list)       # 定义CRF类对象
    max_p, max_y_list = crf.viterbi()                     # 执行维特比算法
    print("max w*F(y, x) = " + str(max_p))                # 打印未归一化最大概率
    max_y_list = np.array(y_list)[max_y_list]             # 将标签下标转换为标签
    print("max y list    = " + str(max_y_list[1:]))       # 打印最大概率标注序列

if __name__ == '__main__':
    main()

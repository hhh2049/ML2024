# encoding=utf-8
import numpy as np

class BFGS:  # 基于Wolfe准则的BFGS优化算法实现
    def __init__(self, fun, grad, x0, K=1e3, tol=1e-6,
                 alpha=1.0, gamma=0.5, c1=0.2, c2=0.9, line_search_J=50):
        self.fun           = fun                             # 待优化函数
        self.grad          = grad                            # 函数的梯度
        self.x0            = x0                              # 迭代的初始值
        self.K             = K                               # 最大迭代次数
        self.tol           = tol                             # 停止迭代阈值
        self.init_alpha    = alpha                           # 参数α，用于线搜索
        self.gamma         = gamma                           # 参数γ，用于线搜索
        self.c1            = c1                              # 参数c1，用于线搜索
        self.c2            = c2                              # 参数c2，用于线搜索
        self.line_search_J = line_search_J                   # 线搜索最大次数

    def optimize(self):  # 执行优化
        fun, grad     = self.fun, self.grad                  # 为缩短代码长度
        gamma, c1, c2 = self.gamma, self.c1, self.c2         # 为缩短代码长度

        xk = self.x0                                         # 第k次迭代的向量xk
        n  = np.shape(self.x0)[0]                            # 获取数据的维度数
        Bk = np.eye(n)                                       # 第k次迭代的矩阵Bk
        k  = 0                                               # 迭代次数计数
        while k < self.K:                                    # 若未达到最大迭代次数
            gk = grad(xk)                                    # 计算当前梯度
            if np.dot(gk, gk) ** 0.5 < self.tol:             # 若梯度小于阈值(接近于0)
                break                                        # 跳出循环，终止优化
            dk = -1.0 * np.linalg.solve(Bk, gk)              # 计算搜索方向(牛顿方向)

            target_j = 0                                     # 符合Wolfe条件的α对应的j
            j = 0                                            # 搜索次数计数
            while j < self.line_search_J:                    # 若未达到最大搜索次数
                a   = gamma ** j * self.init_alpha           # 计算当前的α
                fk1 = fun(xk + a * dk)                       # 计算fun(x_k+1)
                la1 = fun(xk) + c1 * a * np.dot(gk.T, dk)    # 计算l(a1)
                gk1 = grad(xk + a * dk)                      # 计算grad(x_k+1)
                la2 = c2 * np.dot(gk.T, dk)                  # 计算l(a2)

                if fk1 <= la1 and np.dot(gk1.T, dk) >= la2:  # 如果符合wolfe条件
                    target_j = j                             # 记下当前的j
                    break                                    # 跳出循环，终止搜索
                j += 1                                       # 搜索次数加1
            ak = gamma ** target_j * self.init_alpha         # 计算αk值，αk默认为1

            sk = ak * dk                                     # 计算sk
            yk = grad(xk + sk) - gk                          # 计算yk

            ys  = np.dot(yk, sk)                             # 计算相关值，用于更新Bk
            Bs  = np.dot(Bk, sk)                             # 计算相关值，用于更新Bk
            sBs = np.dot(np.dot(sk, Bk), sk)                 # 计算相关值，用于更新Bk
            deltaBk1 = yk.reshape((n, 1)) * yk / ys          # 计算相关值，用于更新Bk
            deltaBk2 = Bs.reshape((n, 1)) * Bs / sBs         # 计算相关值，用于更新Bk
            Bk = Bk + deltaBk1 - deltaBk2                    # 更新Bk

            xk = xk + sk                                     # 更新xk
            k  = k + 1                                       # 迭代计数加1
            print("第" + str(k) + "次的迭代结果为："+str(xk)) # 打印提示信息

        return xk, fun(xk)                                   # 返回极小值点和相应函数值

def main():
    # 函数表达式
    fun  = lambda x: 100*(x[0]**2-x[1])**2 + (x[0]-1)**2
    # 梯度计算式
    grad = lambda x: np.array([400*x[0]*(x[0]**2-x[1])+2*(x[0]-1), -200*(x[0]**2-x[1])])

    bfgs = BFGS(fun, grad, [3, 3])                           # 定义BFGS类对象
    xk, fun_xk = bfgs.optimize()                             # 执行优化
    print("xk =", xk, "f(xk) =", fun_xk)                     # 打印优化结果

if __name__ == '__main__':
    main()

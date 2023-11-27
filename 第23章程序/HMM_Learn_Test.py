# encoding=utf-8
import numpy as np
from HMM_Learn import HMM
from hmmlearn import hmm

def main():
    observation = np.array([[0, 1, 0, 0, 1,
                             0, 0, 0, 1, 1,
                             1, 1, 0, 1, 0,
                             0, 1, 0, 1, 1,
                             0, 0, 0, 1, 0]])                  # 观测序列
    our_hmm = HMM(3, 2, observation[0], n_iter=300, tol=1e-6)  # 定义我们的HMM对象
    our_hmm.train()                                            # 执行训练
    print("our pi:\n%s" % str(our_hmm.pi))                     # 打印初始状态概率向量π
    print("our A: \n%s" % str(our_hmm.A))                      # 打印状态转移概率矩阵A
    print("our B: \n%s" % str(our_hmm.B))                      # 打印观测生成概率矩阵B

    lib_hmm = hmm.CategoricalHMM(n_components=3, n_iter=300,   # 新版HMM，若错误则使用旧版
                                 tol=1e-6, random_state=0)     # 定义官方库HMM对象
    # lib_hmm = hmm.MultinomialHMM(n_components=3, n_iter=300, # 旧版HMM
    #                              tol=1e-6, random_state=0)   # 定义官方库HMM对象
    lib_hmm.fit(observation)                                   # 执行训练
    print("\nlib pi:\n" + str(lib_hmm.startprob_))             # 打印初始状态概率向量π
    print("lib A:\n" + str(lib_hmm.transmat_))                 # 打印状态转移概率矩阵A
    print("lib B:\n" + str(lib_hmm.emissionprob_))             # 打印观测生成概率矩阵B
    print("likelihood:%.6f\n" % lib_hmm.monitor_.history[-1])  # 打印对数似然函数值

if __name__ == "__main__":
    main()

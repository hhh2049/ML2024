# encoding=utf-8
import numpy as np
from HMM_Infer import HMM

def main():
    states       = ["box1", "box2", "box3"]                # 状态名称
    observations = ["black", "white"]                      # 观测名称
    pi = np.array([0.3, 0.5, 0.2])                         # 初始状态概率向量
    A  = np.array([                                        # 状态转移概率矩阵
        [0.4, 0.4, 0.2],
        [0.3, 0.2, 0.5],
        [0.2, 0.6, 0.2]
    ])
    B  = np.array([                                        # 观测生成概率矩阵
        [0.2, 0.8],
        [0.6, 0.4],
        [0.4, 0.6]
    ])
    observation_list = np.array([0, 1, 0])                 # 观测序列(黑，白，黑)

    hmm = HMM(pi, A, B, observation_list)                  # 定义自实现HMM类对象
    p_o_forward  = hmm.forward()                           # 计算前向概率
    p_o_backward = hmm.backward()                          # 计算后向概率
    p_max, i_max_list = hmm.decode()                       # 预测（解码）
    state_list = list(np.array(states)[i_max_list])        # 获取概率最大的状态序列

    print("forward   probability = %f" % p_o_forward)      # 打印前向概率
    print("backward  probability = %f" % p_o_backward)     # 打印后向概率
    print("max       probability = %f" % p_max)            # 打印最大概率
    print("max hidden index list = %s" % str(i_max_list))  # 打印最有可能的索引序列
    print("max hidden state list = %s" % str(state_list))  # 打印最有可能的状态序列

if __name__ == "__main__":
    main()

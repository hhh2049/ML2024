# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt

P_Matrix = np.array([[0.7, 0.2, 0.1],
                     [0.2, 0.6, 0.2],
                     [0.1, 0.4, 0.5]])                      # 状态转移矩阵
pi0_list = np.array([[1/3, 1/3, 1/3],
                     [0.1, 0.1, 0.8],
                     [0.8, 0.1, 0.1]])                      # 三个初始的状态分布
T = 40                                                      # 马尔可夫链序列长度

P = np.eye(3)                                               # 矩阵P初始为单位矩阵
P_list = np.zeros((T, 9))                                   # 存放各个时刻的状态转移矩阵
for i in range(T):                                          # 循环次数：序列长度
    P = np.dot(P, P_Matrix)                                 # 计算两个矩阵的乘积
    P_list[i] = P.reshape(1, 9)                             # 保存各个时刻的状态转移矩阵
print(P, "\n")                                              # 打印矩阵P

plt.figure(figsize=(12, 6))                                 # 创建一个图像并设置画布大小
plt.subplot(1, 2, 1)                                        # 绘制一行两列第一个子图
x = np.arange(T)                                            # 生成一个长度为T的序列
for i in range(9):                                          # 绘制各个矩阵各个元素的曲线
    c = i % 3                                               # 计算该元素属于第几列
    if c == 0:                                              # 如果为矩阵第1列
        plt.plot(x, P_list[:, i],                           # 绘制社会下层概率曲线
                 label="lower"+str(c), linestyle="solid")   # 设置曲线的标签和样式
    if c == 1:                                              # 如果为矩阵第2列
        plt.plot(x, P_list[:, i],                           # 绘制社会中层概率曲线
                 label="middle"+str(c), linestyle="dashed") # 设置曲线的标签和样式
    if c == 2:                                              # 如果为矩阵第3列
        plt.plot(x, P_list[:, i],                           # 绘制社会下层概率曲线
                 label="upper"+str(c), linestyle="dotted")  # 设置曲线的标签和样式
plt.ylim([0.1, 0.6])                                        # 设置第1个子图y轴的坐标范围
plt.legend(loc="upper right", ncol=3)                       # 显示曲线标签，放置于右上角
plt.grid()                                                  # 显示网格

plt.subplot(1, 2, 2)                                        # 绘制一行两列第一个子图
for i in range(len(pi0_list)):                              # 循环次数：初始分布个数
    pi_list = np.zeros((T, 3))                              # 存放各个时刻的状态分布
    pi = pi0_list[i]                                        # 用于存放当前的状态分布
    for j in range(T):                                      # 循环次数：序列长度
        pi = np.dot(pi, P_Matrix)                           # 计算各个时刻的状态分布
        pi_list[j] = pi                                     # 保存各个时刻的状态分布
    print(" " + str(pi))                                    # 打印最后一个状态分布

    plt.plot(x, pi_list[:, 0],                              # 绘制社会下层概率曲线
             label="lower"+str(i), linestyle="solid")       # 设置曲线的标签和样式
    plt.plot(x, pi_list[:, 1],                              # 绘制社会中层概率曲线
             label="middle"+str(i), linestyle="dashed")     # 设置曲线的标签和样式
    plt.plot(x, pi_list[:, 2],                              # 绘制社会上层概率曲线
             label="upper"+str(i), linestyle="dotted")      # 设置曲线的标签和样式
plt.ylim([0.1, 0.6])                                        # 设置第2个子图y轴的坐标范围
plt.legend(loc="upper right", ncol=3)                       # 显示曲线标签，放置于右上角
plt.grid()                                                  # 显示网格
plt.show()                                                  # 显示图像

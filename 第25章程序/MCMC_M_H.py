# encoding=utf-8
import random
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def p(x):  # 目标分布：两个高斯分布之和
    pdf1 = np.exp(-(x - 3.0) ** 2 / 2.0 / 1.0 ** 2) / np.sqrt(2.0 * np.pi) / 1.0
    pdf2 = np.exp(-(x + 1.0) ** 2 / 2.0 / 2.0 ** 2) / np.sqrt(2.0 * np.pi) / 2.0

    return 0.4 * pdf1 + 0.6 * pdf2                   # 返回目标分布的概率密度函数值

n1 = 10000                                           # 燃烧期样本数
n2 = 100000                                          # 需采样样本数
samples    = [0.0] * (n1 + n2)                       # 用于存放所有样本
samples[0] = 1.0                                     # 任选一个起始样本
for t in range(1, n1 + n2):                          # M-H采样
    x      = samples[t - 1]                          # 获取上一个采样样本
    x_star = norm.rvs(loc=x, scale=1, size=1)[0]     # 随机生成一个样本点
    alpha  = min(1, p(x_star) / p(x))                # 计算α值(接受概率)
    u      = random.uniform(0, 1)                    # 采样0~1均匀分布
    if u < alpha:                                    # 如果接受
        samples[t] = x_star                          # 保存新样本
    else:                                            # 如果拒绝
        samples[t] = x                               # 仍用旧样本

x = np.arange(-8, 8, 0.01)                           # 生成一个-8到8步长为0.01的列表
plt.scatter(x, p(x), color="r", label="target")      # 绘制目标分布的散点图
plt.hist(samples[n1:], bins=100, density=True,       # 使用采样的样本集
         edgecolor="b", alpha=0.5, label="sample")   # 绘制采样分布的直方图
plt.legend()                                         # 显示图像的标签
plt.grid()                                           # 显示网格
plt.show()                                           # 显示图像

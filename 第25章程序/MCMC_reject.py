# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

def p(x):  # 目标分布
    pdf1 = np.exp(-(x - 3.0) ** 2 / 2.0 / 1.0 ** 2) / np.sqrt(2.0 * np.pi) / 1.0
    pdf2 = np.exp(-(x + 1.0) ** 2 / 2.0 / 2.0 ** 2) / np.sqrt(2.0 * np.pi) / 2.0

    return 0.4 * pdf1 + 0.6 * pdf2                       # 返回目标分布的概率密度函数值

def q(x):  # 提议分布
    proposal_pdf = proposal_dist.pdf(x)                  # 计算提议分布的概率密度函数值
    return proposal_pdf                                  # 返回提议分布的概率密度函数值

M = 2.0                                                  # 提议分布扩大的倍数
proposal_dist = norm(loc=1.0, scale=3)                   # 定义提议分布（一个高斯分布）
uniform_dist  = uniform(loc=0, scale=1)                  # 定义均匀分布（0～1之间）

def reject_sampling(sample_count=10000):  # 实现拒绝采样算法
    sample_list  = []                                    # 用于存放采样的样本
    sample_total = 0                                     # 用于存放采样的次数

    while len(sample_list) < sample_count:               # 当未达到指定的采样量时
        xi = proposal_dist.rvs(1)[0]                     # 从提议分布生成一个样本
        ui = uniform_dist.rvs(1)[0]                      # 从均匀分布生成一个样本

        if ui < p(xi) / (M * q(xi)):                     # 如果接受
            sample_list.append(xi)                       # 将最新样本存入样本列表

        sample_total += 1                                # 采样的总次数加1

    success_rate = sample_count / sample_total * 100     # 计算拒绝采样的成功率
    success_rate = round(success_rate, 2)                # 保留小数点后两位小数
    print("sampling rate = " + str(success_rate) + "%")  # 打印拒绝采样的成功率

    return sample_list                                   # 返回样本列表

sample_list = reject_sampling(10000)                     # 使用拒绝采样生成10000个样本
plt.figure(figsize=(10, 5))                              # 创建一个图像画布并设置大小
plt.subplot(1, 2, 1)                                     # 绘制一行两列第一个子图
x = np.arange(-10, 10, 0.01)                             # 生成-10到10步长为0.01的列表
plt.plot(x, p(x), color="r",                             # 绘制目标分布的概率密度函数图
         lw=3, alpha=0.5, label="p(x)")                  # 设置曲线的颜色、线宽、透明度
plt.plot(x, M * q(x), color="b", lw=3,                   # 绘制提议分布的概率密度函数图
         alpha=0.5, linestyle="--", label="Mq(x)")       # 设置曲线的透明度、样式等
plt.legend()                                             # 显示曲线的标签
plt.grid()                                               # 显示网格

plt.subplot(1, 2, 2)                                     # 绘制一行两列第二个子图
plt.plot(x, p(x), color="r",                             # 绘制目标分布的概率密度函数图
         lw=3, alpha=0.5, label="p(x)")                  # 设置曲线的颜色、线宽、透明度
plt.hist(sample_list, bins=100, density=True,            # 绘制采样分布的直方图
         edgecolor="b", alpha=0.5, label="sample")       # 设置分组数、是否使用频率等
plt.legend()                                             # 显示图像标签
plt.grid()                                               # 显示网格
plt.show()                                               # 显示图像

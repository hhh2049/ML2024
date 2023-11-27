# encoding=utf-8
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

mean = [5, -1]                                          # 目标分布的均值向量
cov  = [[1, 1], [1, 4]]                                 # 目标分布的协方差矩阵
target_dist = multivariate_normal(mean=mean, cov=cov)   # 目标分布（二维高斯分布）

def p_x1_given_x2(x2, mu1, mu2, sigma1, sigma2):        # 用于计算p(x1|x2)
    mu    = mu1 + rho * sigma1 / sigma2 * (x2 - mu2)    # 计算均值
    sigma = np.sqrt(1 - rho ** 2) * sigma1              # 计算标准差
    x1    = random.normalvariate(mu, sigma)             # 采样得到x1

    return x1                                           # 返回x1

def p_x2_given_x1(x1, mu1, mu2, sigma1, sigma2):        # 用于计算p(x2|x1)
    mu    = mu2 + rho * sigma2 / sigma1 * (x1 - mu1)    # 计算均值
    sigma = np.sqrt(1 - rho ** 2) * sigma2              # 计算标准差
    x2    = random.normalvariate(mu, sigma)             # 采样得到x2

    return x2                                           # 返回x2

n1 = 1000                                               # 燃烧期样本数
n2 = 10000                                              # 需采样样本数
x1_list, x2_list, z_list = [], [], []                   # 用于存放样本
mu1, mu2, sigma1, sigma2, rho = 5, -1, 1, 2, 0.5        # 定义二维高斯分布参数
x2 = mu2                                                # 定义x2的初始值

for i in range(n1+n2):                                  # 生成n1+n2个样本
    x1 = p_x1_given_x2(x2, mu1, mu2, sigma1, sigma2)    # 基于x2采样得到x1
    x2 = p_x2_given_x1(x1, mu1, mu2, sigma1, sigma2)    # 基于x1采样得到x2
    z  = target_dist.pdf([x1, x2])                      # 计算二维分布的概率密度函数值
    x1_list.append(x1)                                  # 将x1加入x1样本列表
    x2_list.append(x2)                                  # 将x2加入x2样本列表
    z_list.append(z)                                    # 将z加入z样本列表

plt.hist(x1_list[n1:], bins=50, density=True,           # 使用x1轴样本集
         edgecolor='g', alpha=0.5, label='x1')          # 绘制x1轴直方图
plt.hist(x2_list[n1:], bins=50, density=True,           # 使用x2轴样本集
         edgecolor='r', alpha=0.5, label='x2')          # 绘制x2轴直方图
plt.legend()                                            # 显示图像标签
plt.grid()                                              # 显示网格
plt.show()                                              # 显示图像

x1, x2, z = x1_list, x2_list, z_list                    # 为缩短代码长度
figure = plt.figure()                                   # 创建一个图像画布
ax = Axes3D(figure, azim=20)                            # 创建一个3D图像
ax.scatter(x1[n1:], x2[n1:], z[n1:], marker='o')        # 绘制3D散点图
figure.add_axes(ax, label="3D")                         # 在画布中加入图像
plt.show()                                              # 显示图像

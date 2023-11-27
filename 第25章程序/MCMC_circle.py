# encoding=utf-8
import random
import numpy as np
import matplotlib.pyplot as plt

n_point = 100000                                          # 随机生成的点的数量
x_list  = [0.0] * n_point                                 # 存放x轴坐标
y_list  = [0.0] * n_point                                 # 存放y轴坐标

for i in range(n_point):                                  # 利用均匀分布随机生成10万个点
    x_list[i] = random.uniform(0, 1)                      # 随机生成x轴坐标
    y_list[i] = random.uniform(0, 1)                      # 随机生成y轴坐标

n_in_circle = 0                                           # 统计有多少个点在圆内
for i in range(n_point):                                  # 遍历所有的点
    r = np.sqrt((x_list[i]-0.5)**2 + (y_list[i]-0.5)**2)  # 计算当前点到圆心的距离
    if r <= 0.5:                                          # 如果当前点在圆内
        n_in_circle += 1                                  # 圆内点的计数加1

area_real      = np.pi * 0.5 ** 2                         # 计算真实的圆面积
area_estimated = n_in_circle / n_point * 1.0              # 利用随机数计算圆面积
pi_estimated   = area_estimated / 0.5 / 0.5               # 利用随机数估计π的值
area_real      = round(area_real, 4)                      # 保留4位小数
area_estimated = round(area_estimated, 4)                 # 保留4位小数
pi_estimated   = round(pi_estimated, 4)                   # 保留4位小数

print("     real area: " + str(area_real))                # 打印真实的圆面积
print("estimated area: " + str(area_estimated))           # 打印利用随机数计算的圆面积
print("  estimated pi: " + str(pi_estimated))             # 打印利用随机数估计的π值

plt.plot(x_list, y_list, "go", alpha=0.2, markersize=0.3) # 绘制散点图并设置相关样式
plt.axis("equal")                                         # 设置x、y轴刻度等长
circle = plt.Circle(xy=(0.5, 0.5), radius=0.5, alpha=0.5) # 按圆心半径透明度绘制一个圆
plt.gca().add_patch(circle)                               # 将圆加入到图像中
plt.grid()                                                # 显示网格
plt.show()                                                # 显示图像

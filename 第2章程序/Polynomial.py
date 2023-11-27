# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

x_train = np.random.uniform(0.0, 1.0, 25)                     # 从0.0到1.0均匀采样25个数
f_x     = np.cos(1.5 * np.pi * x_train)                       # 计算真实值
y_train = f_x + np.random.randn(25) * 0.1                     # 在真实值中加入噪声
X_train = np.c_[x_train, x_train**2, x_train**3, x_train**4]  # 转换为多项式输入(维度25×4)

lr = LinearRegression(X_train, y_train, fit_type=0)           # 定义线性回归类对象
lr.fit()                                                      # 拟合数据，训练模型

x_test = np.linspace(0.0, 1.0, 50)                            # 从0.0到1.0均匀采样50个数
X_test = np.c_[x_test, x_test**2, x_test**3, x_test**4]       # 转换为多项式输入(维度50×4)
y_test = lr.predict(X_test)                                   # 使用模型计算预测值

plt.plot(x_train, y_train, "o")                               # 绘制原始训练数据点
plt.plot(x_test, y_test, color="r")                           # 绘制拟合数据的曲线
plt.subplots_adjust(top=0.98, bottom=0.05, left=0.05,
                    right=0.98, hspace=0, wspace=0)           # 调整图像边界
plt.savefig("polynomial.png")                                 # 将图像保存到当前目录
plt.show()                                                    # 显示图像

# encoding=utf-8
import numpy as np
import time

# 使用线性同余法，生成指定数量和指定范围内的随机数
def random_number(number_count, number_scope=10, random_state=None):
    a, b, m = 7 ** 5, 0, 2 ** 31 - 1                         # 定义用于生成随机数的变量

    xi = random_state                                        # 初始的迭代变量
    if random_state is None:                                 # 若为None，使用系统时钟
        now = time.time()                                    # 获取当前时间（一个实数）
        xi = int(now * 1e5)                                  # 乘以1e5并取整

    number_list = []                                         # 用于存放生成的随机数
    for i in range(number_count):                            # 生成指定数量的随机数
        xi = (a * xi + b) % m                                # 使用线性同余法
        number_list.append(xi % number_scope)                # 转化为0～9之间的数字

    return number_list                                       # 返回生成的随机数列表

# 在产生的随机数中，统计各个数字的数量和比例
def get_digit_count(numbers):
    digits, counts = np.unique(numbers, return_counts=True)  # 统计各个数字的数量
    ratios = counts / np.sum(counts)                         # 计算各个数字的比例
    ratios = np.round(ratios, 2)                             # 保留2位小数

    return digits, counts, ratios                            # 返回数字、数量、比例

numbers_list = random_number(1000000, 10)                    # 随机生成100万个0~9的数字
digits, counts, ratios = get_digit_count(numbers_list)       # 统计各个数字的数量和占比
print(digits)                                                # 打印各个数字
print(counts)                                                # 打印各个数字的数量
print(ratios)                                                # 打印各个数字的比例

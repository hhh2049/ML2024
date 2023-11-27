# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from SVD import SVD

# 按照奇异值总和或数量的百分比压缩图像数据
def compress_image_using_SVD(data, percent, percent_type=1):
    our_svd = SVD(data, n_components=percent)               # 定义自实现SVD类对象
    our_svd.fit_transform()                                 # 执行训练与数据转换
    U, sigma, V_T = our_svd.U, our_svd.sigma_, our_svd.V_T  # 获取奇异值分解结果
    Sigma = np.diag(sigma)                                  # 将奇异值向量转换为方阵
    # U, sigma, V_T = np.linalg.svd(data)                   # 使用官方库进行奇异值分解
    # Sigma = np.diag(sigma)                                # 将奇异值向量转换为方阵

    k = 0                                                   # 用于存储降维后的维度数
    if percent_type == 1:                                   # 按奇异值总和百分比确定k
        eigen_sum = 0.0                                     # 当前的奇异值之和
        threshold = np.sum(sigma) * percent                 # 由百分比计算奇异值阈值
        for i in range(len(sigma)):                         # 遍历所有的奇异值
            eigen_sum += sigma[i]                           # 计算当前奇异值之和
            if round(eigen_sum, 2) >= round(threshold, 2):  # 当前奇异值之和超过阈值
                k = i + 1                                   # 确定k值
                break                                       # 中止循环
    else:                                                   # 按奇异值数量百分比确定k
        k = int(len(sigma) * percent)                       # 计算k值

    Data = np.dot(np.dot(U[:,:k], Sigma[:k,:k]), V_T[:k,:]) # 压缩图像数据
    Data[Data < 0] = 0                                      # 像素值若小于0则用0替代
    Data[Data > 255] = 255                                  # 像素值若大于255则用255替代

    return np.rint(Data).astype("uint8")                    # 将数据按四舍五入取整并返回

# 读取、压缩和保存图像
def compress_image(filename, percent, type):
    image = plt.imread(filename, "r")                       # 读取图像数据
    image = np.array(image, dtype="int32")                  # 转换为numpy数组
    Red   = image[:, :, 0]                                  # 获取红色通道数据
    Green = image[:, :, 1]                                  # 获取绿色通道数据
    Blue  = image[:, :, 2]                                  # 获取蓝色通道数据

    Red   = compress_image_using_SVD(Red, percent, type)    # 压缩红色通道数据
    Green = compress_image_using_SVD(Green, percent, type)  # 压缩绿色通道数据
    Blue  = compress_image_using_SVD(Blue, percent, type)   # 压缩蓝色通道数据

    image = np.stack((Red, Green, Blue), 2)                 # 重建图像
    # Image.fromarray(image).show()                         # 显示压缩后的图像
    filename = "D:/"+str(type)+"_" + str(percent) + ".jpg"  # 生成文件名，保存到D盘根目录
    Image.fromarray(image).save(filename)                   # 保存图像

def main():
    filename = "D:/test.jpg"                                # 图像存放于D盘根目录下
    percent_list1 = [0.25, 0.5, 0.75, 0.85, 1.0]            # 定义百分比数组
    for percent in percent_list1:                           # 遍历所有百分比
        compress_image(filename, percent, type=1)           # 按奇异值总和百分比压缩图像
    percent_list2 = [0.01, 0.05, 0.1, 0.25, 1.0]            # 定义百分比数组
    for percent in percent_list2:                           # 遍历所有百分比
        compress_image(filename, percent, type=2)           # 按奇异值数量百分比压缩图像

if __name__ == "__main__":
    main()

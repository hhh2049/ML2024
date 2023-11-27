# encoding=utf-8
import numpy as np
from Ridge import Ridge
from ml_tools import standardize_data
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge as SKL_Ridge

def main():
    # 使用sklearn包中的波士顿房价数据，预测波士顿房价
    boston_house = load_boston()                                  # 导入波士顿房价数据
    X, y = boston_house.data, boston_house.target                 # 获取数据集和真实值
    X, y = standardize_data(X, y)                                 # 数据标准化
    Z = train_test_split(X, y, test_size=0.2, random_state=2)     # 划分训练集和测试集
    X_train, X_test, y_train, y_test = Z[0], Z[1], Z[2], Z[3]     # 训练集和测试集赋值

    # 测试自实现Ridge回归算法（使用最小二乘法）
    our_ridge = Ridge(X_train, y_train, lamda=0.1, fit_type=0)    # 定义自实现类对象
    our_ridge.fit()                                               # 训练模型
    our_y_predict = our_ridge.predict(X_test)                     # 执行预测
    our_mse = mean_squared_error(y_test, our_y_predict)           # 计算均方误差
    print("our own ridge MSE   = %f" % our_mse)                   # 打印均方误差
    my_score = our_ridge.score(X_test, y_test)                    # 计算预测得分
    print("our own ridge score = %f" % my_score)                  # 打印预测得分

    # 测试官方库Ridge回归算法（使用最小二乘法）
    skl_ridge = SKL_Ridge(alpha=0.1, solver="svd")                # 定义官方库类对象
    skl_ridge.fit(X_train, y_train)                               # 训练模型
    skl_y_predict = skl_ridge.predict(X_test)                     # 执行预测
    skl_ridge_mse = mean_squared_error(y_test, skl_y_predict)     # 计算均方误差
    print("sklearn ridge MSE   = %f" % skl_ridge_mse)             # 打印均方误差
    skl_score = skl_ridge.score(X_test, y_test)                   # 计算预测得分
    print("sklearn ridge score = %f\n" % skl_score)               # 打印预测得分

    # 测试自实现Ridge回归算法（使用梯度下降法, λ=0.1，使用正则化）
    our_ridge = Ridge(X_train, y_train, lamda=0.1, fit_type=1)    # 定义自实现类对象
    our_ridge.fit()                                               # 训练模型
    our_y_predict = our_ridge.predict(X_test)                     # 执行预测
    our_mse = mean_squared_error(y_test, our_y_predict)           # 计算均方误差
    print("our own ridge MSE   = %f (λ=0.1)" % our_mse)           # 打印均方误差
    our_score = our_ridge.score(X_test, y_test)                   # 计算预测得分
    print("our own ridge score = %f (λ=0.1)" % our_score)         # 打印预测得分
    print("w L2 = %f" % np.dot(our_ridge.w, our_ridge.w) ** 0.5)  # 打印w的L2范数

    # 测试自实现Ridge回归算法（使用梯度下降法, λ=0.0，未使用正则化）
    our_ridge = Ridge(X_train, y_train, lamda=0, fit_type=1)      # 定义自实现类对象
    our_ridge.fit()                                               # 训练模型
    our_y_predict = our_ridge.predict(X_test)                     # 执行预测
    our_mse = mean_squared_error(y_test, our_y_predict)           # 计算均方误差
    print("our own ridge MSE   = %f (λ=0.0)" % our_mse)           # 打印均方误差
    our_score = our_ridge.score(X_test, y_test)                   # 计算预测得分
    print("our own ridge score = %f (λ=0.0)" % our_score)         # 打印预测得分
    print("w L2 = %f" % np.dot(our_ridge.w, our_ridge.w) ** 0.5)  # 打印w的L2范数

if __name__ == "__main__":
    main()

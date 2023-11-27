# encoding=utf-8
import numpy as np
from ml_tools import standardize_data
from Lasso import LassoRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, SGDRegressor

def main():
    # 使用sklearn包中的波士顿房价数据（在sklearn 1.2中将被删除），预测波士顿房价
    boston_house = load_boston()                                # 导入波士顿房价数据
    X, y = boston_house.data, boston_house.target               # 获取数据集和真实值
    X, y = standardize_data(X, y)                               # 数据标准化
    Z = train_test_split(X, y, test_size=0.2, random_state=2)   # 划分训练集和测试集
    X_train, X_test, y_train, y_test = Z[0], Z[1], Z[2], Z[3]   # 训练集和测试集赋值

    # 测试基于坐标下降法的线性回归算法，不使用正则化
    our_lasso = LassoRegression(X_train, y_train, fit_type=0)   # 定义自实现类对象
    our_lasso.fit()                                             # 训练模型
    our_y_predict = our_lasso.predict(X_test)                   # 执行预测
    our_mse = mean_squared_error(y_test, our_y_predict)         # 计算均方误差
    print("our own Linear Regression MSE   = %f" % our_mse)     # 打印均方误差
    our_score = our_lasso.score(X_test, y_test)                 # 计算预测得分
    print("our own Linear Regression score = %f\n" % our_score) # 打印预测得分

    # 测试官方库的线性回归算法，不使用正则化
    skl_sgdRegressor = SGDRegressor(alpha=0.0)                  # 定义官方库类对象
    skl_sgdRegressor.fit(X_train, y_train)                      # 训练模型
    skl_sgd_y_predict = skl_sgdRegressor.predict(X_test)        # 执行预测
    skl_mse = mean_squared_error(y_test, skl_sgd_y_predict)     # 计算均方误差
    print("sklearn SGDRegressor MSE        = %f" % skl_mse)     # 打印均方误差
    skl_score = skl_sgdRegressor.score(X_test, y_test)          # 计算预测得分
    print("sklearn SGDRegressor score      = %f\n" % skl_score) # 打印预测得分

    # 以下为自实现Lasso的运行情况
    our_lasso = LassoRegression(X_train, y_train, lamda=0.1)    # 定义自实现类对象
    our_lasso.fit()                                             # 训练模型
    our_y_predict = our_lasso.predict(X_test)                   # 执行预测
    our_mse = mean_squared_error(y_test, our_y_predict)         # 计算均方误差
    print("our own lasso MSE   = %f" % our_mse)                 # 打印均方误差
    our_score = our_lasso.score(X_test, y_test)                 # 计算预测得分
    print("our own lasso score = %f" % our_score)               # 打印预测得分
    L2_w_b = np.dot(our_lasso.w, our_lasso.w) ** 0.5            # 计算模型参数的L2范数
    print("our own w L2        = %f" % L2_w_b)                  # 打印模型参数的L2范数
    print(our_lasso.w[1:], "\n")                                # 打印权重向量w

    # 以下为官方库Lasso的运行情况
    skl_lasso = Lasso(alpha=0.1, fit_intercept=False)           # 定义官方库类对象
    skl_lasso.fit(X_train, y_train)                             # 训练模型
    skl_y_predict = skl_lasso.predict(X_test)                   # 执行预测
    skl_lasso_mse = mean_squared_error(y_test, skl_y_predict)   # 计算均方误差
    print("sklearn Lasso MSE   = %f" % skl_lasso_mse)           # 打印均方误差
    skl_score = skl_lasso.score(X_test, y_test)                 # 计算预测得分
    print("sklearn lasso score = %f" % skl_score)               # 打印预测得分
    skl_w_b = np.append(skl_lasso.coef_, skl_lasso.intercept_)  # 连接权重w和偏置b
    L2_w_b = np.dot(skl_w_b, skl_w_b) ** 0.5                    # 计算模型参数的L2范数
    print("sklearn w L2        = %f" % L2_w_b)                  # 打印模型参数的L2范数
    print(skl_lasso.coef_)                                      # 打印权重向量w

if __name__ == "__main__":
    main()

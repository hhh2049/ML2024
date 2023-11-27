# encoding=utf-8
from LinearRegression import LinearRegression as OUR_LR          # 自实现线性回归类
from ml_tools import normalize_data                              # 自实现工具函数
from sklearn.datasets import load_boston                         # 波士顿房价数据集
from sklearn.model_selection import train_test_split             # 数据集切分工具函数
from sklearn.linear_model import LinearRegression as SKL_LR      # 官方库线性回归类1
from sklearn.linear_model import SGDRegressor                    # 官方库线性回归类2
from sklearn.metrics import mean_squared_error                   # 官方库计算均方误差
import matplotlib.pyplot as plt                                  # 导入绘图模块

def main():  # 对比测试自实现线性回归和官方库线性回归
    # 使用sklearn包中自带的波士顿房价数据集，预测波士顿房价
    boston_house = load_boston()                                 # 导入波士顿房价数据
    X, y = boston_house.data, boston_house.target                # 获取数据集和真实值
    X, y = normalize_data(X, y)                                  # 预处理数据，对数据归一化
    X_train, X_test, y_train, y_test = train_test_split(X, y,    # 切分数据集
                                       test_size=0.25, random_state=33)

    # 使用官方库的线性回归类（基于最小二乘法）
    skl_lr = SKL_LR()                                            # 定义官方库线性回归类对象
    skl_lr.fit(X_train, y_train)                                 # 训练模型
    skl_lr_y_predict = skl_lr.predict(X_test)                    # 执行预测
    skl_lr_mse = mean_squared_error(y_test, skl_lr_y_predict)    # 计算均方误差
    print("skl LinearRegression MSE   = %f" % skl_lr_mse)        # 打印均方误差
    skl_lr_score = skl_lr.score(X_test, y_test)                  # 计算预测得分
    print("skl LinearRegression score = %f" % skl_lr_score)      # 打印预测得分

    # 使用自实现的线性回归类（基于最小二乘法）
    our_lr = OUR_LR(X_train, y_train, fit_type=0)                # 定义自实现线性回归类对象
    our_lr.fit()                                                 # 训练模型
    our_lr_y_predict = our_lr.predict(X_test)                    # 执行预测
    our_lr_mse = mean_squared_error(y_test, our_lr_y_predict)    # 计算均方误差
    print("our LinearRegression MSE   = %f" % our_lr_mse)        # 打印方误差
    our_lr_score = our_lr.score(X_test, y_test)                  # 计算预测得分
    print("our LinearRegression score = %f\n" % our_lr_score)    # 打印预测得分

    # 使用官方库的线性回归类（基于梯度下降法）
    skl_sgdRegressor = SGDRegressor(alpha=0.0)                   # 定义官方库线性回归类对象
    skl_sgdRegressor.fit(X_train, y_train)                       # 训练模型
    skl_sgd_y_predict = skl_sgdRegressor.predict(X_test)         # 执行预测
    skl_sgd_mse = mean_squared_error(y_test, skl_sgd_y_predict)  # 计算均方误差
    print("skl SGDRegressor MSE       = %f" % skl_sgd_mse)       # 打印方误差
    skl_sgd_score = skl_sgdRegressor.score(X_test, y_test)       # 计算预测得分
    print("skl SGDRegressor score     = %f" % skl_sgd_score)     # 打印预测得分

    # 使用自实现的线性回归类（基于梯度下降法）
    our_lr = OUR_LR(X_train, y_train, fit_type=1)                # 定义自实现线性回归类对象
    our_lr.fit()                                                 # 训练模型
    our_lr_y_predict = our_lr.predict(X_test)                    # 执行预测
    our_lr_mse = mean_squared_error(y_test, our_lr_y_predict)    # 计算均方误差
    print("our LinearRegression MSE   = %f" % our_lr_mse)        # 打印方误差
    our_lr_score =our_lr.score(X_test, y_test)                   # 计算预测得分
    print("our LinearRegression score = %f" % our_lr_score)      # 打印预测得分

    # 测试不同学习率对算法执行效果的影响
    eta = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.4]   # 定义不同的学习率
    score_list = []                                              # 存放预测得分的列表
    for i in range(len(eta)):                                    # 使用不同的学习率
        lr = OUR_LR(X_train, y_train, fit_type=1, eta=eta[i])    # 定义自实现线性回归类对象
        lr.fit()                                                 # 训练模型
        score = lr.score(X_test, y_test)                         # 计算预测得分
        score_list.append(score)                                 # 将得分加入列表
    for i in range(len(eta)):                                    # 打印得分的列表
        print("%f " % score_list[i], end="")

    # 绘制图像
    figure = plt.figure()                                        # 定义图像
    axis = figure.add_subplot(1, 1, 1)                           # 添加1行1列第1个子图
    axis.plot(eta, score_list, "s-")                             # 定义线的样式
    axis.set_xscale("log")                                       # x轴尺度定义为log(x)
    axis.set_title("LinearRegression")                           # 设置图像标题
    axis.set_xlabel("eta")                                       # 设置x轴标题
    axis.set_ylabel("score")                                     # 设置y轴标题
    plt.show()                                                   # 显示图像

if __name__ == "__main__":
    main()

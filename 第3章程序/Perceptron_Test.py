# encoding=utf-8
from Perceptron_Primal import Perceptron_Primal            # 导入自实现原始形式感知机
from Perceptron_Dual import Perceptron_Dual                # 导入自实现对偶形式感知机
from Perceptron_Error import Perceptron_Error              # 导入自实现误差形式感知机
from sklearn.linear_model import Perceptron                # 导入官方库感知机
from sklearn.datasets import load_iris                     # 导入官方库鸢尾花数据集

def main():
    iris = load_iris()                                     # 载入鸢尾花数据集
    X, y = iris.data, iris.target                          # 获取数据和真实值标签
    y[:50] = 1                                             # 前50条数据赋值1
    y[50:150] = -1                                         # 后100条数据赋值-1

    skl_pt = Perceptron()                                  # 定义官方库感知机
    skl_pt.fit(X, y)                                       # 拟合数据，训练模型
    skl_score = skl_pt.score(X, y)                         # 计算训练得分
    print("sklearn Perceptron score    = %f" % skl_score)  # 打印训练得分

    our_pt1 = Perceptron_Primal(X, y)                      # 定义自实现原始形式感知机
    our_pt1.fit()                                          # 拟合数据，训练模型
    our_score1 = our_pt1.score(X, y)                       # 计算训练得分
    print("our Perceptron_Primal score = %f" % our_score1) # 打印训练得分

    our_pt2 = Perceptron_Dual(X, y)                        # 定义自实现对偶形式感知机
    our_pt2.fit()                                          # 拟合数据，训练模型
    our_score2 = our_pt2.score(X, y)                       # 计算训练得分
    print("our Perceptron_Dual score   = %f" % our_score2) # 打印训练得分

    y[50:150] = 0                                          # 后100条数据赋值0
    our_pt3 = Perceptron_Error(X, y)                       # 定义自实现误差形式感知机
    our_pt3.fit()                                          # 拟合数据，训练模型
    our_score3 = our_pt3.score(X, y)                       # 计算训练得分
    print("our Perceptron_Error score  = %f" % our_score3) # 打印训练得分

if __name__ == "__main__":
    main()

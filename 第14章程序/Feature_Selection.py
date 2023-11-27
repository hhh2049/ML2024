# encoding=utf-8
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, f_regression
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_digits, load_diabetes, load_breast_cancer

# 过滤式特征选择之一：通过特征方差过滤特征
X = [[100, 1, 2, 3],
     [101, 4, 5, 6],
     [102, 7, 8, 9]]                             # 定义训练数据集
selector = VarianceThreshold(threshold=1.0)      # 方差过滤，阈值为1.0
selector.fit(X)                                  # 执行特征过滤
print(selector.variances_)                       # 打印各个特征的方差
print(selector.get_support(indices=False))       # 打印特征过滤的结果
print(selector.transform(X), "\n")               # 打印特征过滤后的数据

# 过滤式特征选择之二：通过统计指标过滤特征
X, y     = load_digits(return_X_y=True)          # 载入手写数字分类数据集
selector = SelectKBest(chi2, k=20)               # 卡方过滤，保留20个特征
X_       = selector.fit_transform(X, y)          # 执行特征过滤和数据转换
print(X.shape, X_.shape)                         # 打印特征过滤前后的数据维度
print(selector.get_support(indices=True))        # 打印被保留的特征的下标

X, y     = load_diabetes(return_X_y=True)        # 载入糖尿病回归数据集
selector = SelectPercentile(f_regression,        # f回归过滤
                            percentile=60)       # 保留前60%的特征
X_       = selector.fit_transform(X, y)          # 执行特征过滤和数据转换
print(X.shape, X_.shape)                         # 打印特征过滤前后的数据维度
print(selector.get_support(indices=True))        # 打印被选中的特征的下标
print(selector.scores_, "\n")                    # 打印所有特征的得分

# 包裹式特征选择
X, y      = load_breast_cancer(return_X_y=True)  # 载入乳腺癌分类数据集
estimator = SVC(kernel="linear", C=1)            # 定义一个外部学习器(支持向量机)
selector  = RFE(estimator=estimator,             # 定义特征选择的类对象
                n_features_to_select=15)         # 保留15个特征
X_        = selector.fit_transform(X, y)         # 执行特征选择和数据转换
print(X.shape, X_.shape)                         # 打印特征选择前后的数据维度
print(selector.n_features_)                      # 打印被保留的特征的数量
print(selector.support_)                         # 打印特征选择的结果
print(selector.ranking_, "\n")                   # 打印特征排名(1表示被保留的特征)

# 嵌入式特征选择之一：使用线性模型
X, y      = load_breast_cancer(return_X_y=True)  # 载入乳腺癌分类数据集
estimator = LogisticRegression(penalty='l1',     # 定义一个外部学习器(逻辑回归)
                               solver="saga",    # 使用L1正则化和SAGA优化算法
                               max_iter=5000)    # 设置最大训练次数
selector  = SelectFromModel(estimator=estimator, # 定义特征选择的类对象
                            threshold="median")  # 使用中位数作为阈值
X_        = selector.fit_transform(X, y)         # 执行特征选择和数据转换
print(X.shape, X_.shape)                         # 打印特征选择前后的数据维度
print(selector.threshold_)                       # 打印特征选择的阈值
print(selector.get_support(True), "\n")          # 打印被保留的特征的下标

# 嵌入式特征选择之二：使用决策树相关模型
X, y      = load_breast_cancer(return_X_y=True)  # 载入乳腺癌分类数据集
estimator = GradientBoostingClassifier()         # 定义一个外部学习器(梯度提升树)
selector  = SelectFromModel(estimator=estimator, # 定义特征选择的类对象
                            threshold="median")  # 使用中位数作为阈值
X_        = selector.fit_transform(X, y)         # 执行特征选择和数据转换
print(X.shape, X_.shape)                         # 打印特征选择前后的数据维度
print(selector.threshold_)                       # 打印特征选择的阈值
print(selector.get_support(True))                # 打印被保留的特征的下标

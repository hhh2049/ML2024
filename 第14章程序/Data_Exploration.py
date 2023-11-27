# encoding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 使用pandas导入爱荷华州艾姆斯房价数据集
train = pd.read_csv("D:/HousePrice/train.csv")
test  = pd.read_csv("D:/HousePrice/test.csv")
# 使用pandas导入泰坦尼克号乘客幸存数据集
# train = pd.read_csv("D:/Titanic/train.csv")
# test  = pd.read_csv("D:/Titanic/test.csv")

# 打印训练集的基本信息、描述信息、前五行
print(train.info())
print(train.describe())
print(train.head())
# 打印测试集的基本信息、描述信息、前十行
# print(test.info())
# print(test.describe())
# print(test.head(10))

# 统计各列缺失值情况
# rows = []
# for i in train.columns:
#     rows.append((i,
#                  train[i].nunique(),
#                  train[i].isnull().sum() * 100 / train.shape[0],
#                  train[i].dtype))
# missing_df = pd.DataFrame(rows, columns=["Column",
#                                          "Unique Values",
#                                          "Percentage of Missing",
#                                          "Type"])
# print(missing_df.sort_values("Percentage of Missing", ascending=False)[:10])
# # 图形化显示缺失值情况
# missing = train.isnull().sum()
# missing = missing[missing > 0]
# missing.sort_values(inplace=True)
# missing.plot.bar()
# plt.subplots_adjust(top=0.95, bottom=0.25, left=0.1, right=0.95)
# plt.show()

# 绘制房价数据集的箱线图
# fig = plt.figure(figsize=(18, 9))
# plt.subplot(1, 3, 1)
# sns.boxplot(y=train["MSSubClass"], orient="v", width=0.5)
# plt.subplot(1, 3, 2)
# sns.boxplot(y=train["LotFrontage"], orient="v", width=0.5)
# plt.subplot(1, 3, 3)
# sns.boxplot(y=train["LotArea"], orient="v", width=0.5)
# plt.subplots_adjust(top=0.95, bottom=0.25, left=0.1, right=0.95)
# plt.show()

# 单变量分析之一
# fig = plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# sns.distplot(train["LotArea"], fit=stats.norm)  # LotArea可替换为标签SalePrice
# plt.subplot(1, 2, 2)
# stats.probplot(train["LotArea"], plot=plt)  # LotArea可替换为标签SalePrice
# plt.show()

# 单变量分析之二
# fig = plt.figure(figsize=(12, 6))
# ax = sns.kdeplot(train["LotFrontage"], color="Red",  shade=True)
# ax = sns.kdeplot(test["LotFrontage"],  color="Blue", shade=True)
# ax.set_xlabel("LotFrontage")
# ax.set_ylabel("Frequency")
# ax.legend(["train", "test"])
# plt.show()

# 多变量分析之一
# fig = plt.figure(figsize=(12, 6))
# ax = plt.subplot(1, 2, 1)
# sns.regplot(x="GrLivArea", y="SalePrice", data=train, ax=ax,
#             scatter_kws={"marker": ".", "s":3, "alpha": 0.3},
#             line_kws={"color": "k"})
# plt.xlabel("GrLivArea")
# plt.ylabel("target")
# ax = plt.subplot(1, 2, 2)
# sns.distplot(train["GrLivArea"].dropna())
# plt.xlabel("GrLivArea")
# plt.show()

# 多变量分析之二
# train_corr = train.corr()
# plt.subplots(figsize=(18, 8))
# sns.heatmap(train_corr, vmax=0.5, square=True)
# plt.subplots_adjust(top=0.95, bottom=0.15, left=0.05, right=0.95)
# plt.show()

# 多变量分析之三
# top = 10
# train_corr = train.corr()
# columns = train_corr.nlargest(top, "SalePrice")["SalePrice"].index
# np.corrcoef(train[columns].values.T)
# plt.subplots(figsize=(18, 8))
# sns.heatmap(train[columns].corr(), vmax=0.8, square=True, annot=True)
# plt.subplots_adjust(top=0.95, bottom=0.15, left=0.05, right=0.95)
# plt.show()

# 通过Box-Cox转换，将特征转为正态分布
# def scale_min_max(col):  # 特征归一化
#     return (col - col.min()) / (col.max() - col.min())
#
# old_feature    = train["GrLivArea"]
# new_feature, _ = stats.boxcox(old_feature)
# new_feature    = scale_min_max(new_feature)
#
# fig = plt.figure(figsize=(18, 8))
# plt.subplot(1, 2, 1)
# sns.distplot(new_feature, fit=stats.norm)  # 可以使用old_feature
# plt.subplot(1, 2, 2)
# stats.probplot(new_feature, plot=plt)  # 可以使用old_feature
# plt.show()

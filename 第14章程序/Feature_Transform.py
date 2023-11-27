# encoding=utf-8
from sklearn import preprocessing

# 标签编码
encoder = preprocessing.LabelEncoder()
encoder.fit([1, 2, 2, 6])
print(encoder.transform([1, 1, 2, 6]))
print(encoder.inverse_transform([0, 0, 1, 2]))
encoder.fit(["paris", "paris", "tokyo", "beijing"])
print(encoder.transform(["paris", "tokyo", "beijing"]), "\n")

# 独热编码
encoder = preprocessing.OneHotEncoder(handle_unknown="ignore")
X = [['Male', 1], ['Female', 3], ['Female', 2]]
encoder.fit(X)
print(encoder.transform([['Female', 1], ['Male', 4]]).toarray(), "\n")

# 训练数据
X = [[1, -1,  2],
     [2,  0,  0],
     [0,  1, -1]]

# 标准化
scaler = preprocessing.StandardScaler()
scaler.fit(X)
print(scaler.mean_)
print(scaler.scale_)
print(scaler.transform(X), "\n")

# 归一化
scaler = preprocessing.MinMaxScaler()
print(scaler.fit_transform(X), "\n")

# 二值化
binarizer = preprocessing.Binarizer(threshold=1.0)
print(binarizer.transform(X), "\n")

# 正则化
scaler = preprocessing.Normalizer(norm="l2")
scaler.fit(X)
print(scaler.transform(X))

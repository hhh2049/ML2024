# encoding=utf-8
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# 缺失值填充示例
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
X_train = [[1, 2], [np.nan, 3], [7, 10]]
imp.fit(X_train)
X_test = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X_test))

X_df = pd.DataFrame([["a",    "x"],
                     [np.nan, "y"],
                     ["a",    np.nan],
                     ["b",    "y"]], dtype="category")
imp = SimpleImputer(strategy="most_frequent")
print(imp.fit_transform(X_df))

# 异常值示例
x = [5, 5, 8, 8, 8, 8, 10, 10, 10, 15]
print(np.mean(x))
print(np.std(x))
print(np.median(x))
x = np.array([5, 5, 8, 8, 8, 8, 10, 10, 10, 15, 200])
print(np.mean(x))
print(np.std(x))
print(np.median(x))

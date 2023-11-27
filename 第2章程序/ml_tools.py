# encoding=utf-8
import numpy as np

# 数据预处理：对数据进行标准化
# 参数X_matrix：训练或测试数据，维度(m×n)，m行n列，共m个数据，每个数据有n个特征
# 参数y_vector：真实值，维度(m,)，m行1列，共m个真实值
def standardize_data(X_matrix, y_vector=None):
    X = X_matrix.copy()                            # 复制矩阵，防止破坏原始数据集

    m, n = X.shape                                 # 获取数据集的数据量m和特征数n
    for j in range(n):                             # 遍历数据集的每个特征
        mean = np.mean(X[:, j])                    # 获取该特征（第j列）的平均值
        std = np.std(X[:, j])                      # 获取该特征（第j列）的标准差
        if std == 0:                               # 若标准差为0，说明所有值都相等
            X[:, j] = 0                            # 将该列所有数据设为0
        else:                                      # 对该特征（第j列）每个值标准化
            for i in range(m):                     # 为增加代码的可读性，此处使用循环
                X[i][j] = (X[i][j] - mean) / std   # 对第i个数据的第j个特征执行标准化

    if y_vector is None:                           # 若y_vector为空，只对X进行标准化
        return X, None                             # 返回标准化后的数据集

    y = y_vector.copy()                            # 复制列向量，防止破坏原始的真实值
    mean = np.mean(y)                              # 获取真实值的平均值
    std = np.std(y)                                # 获取真实值的标准差
    if std == 0: return X, y                       # 若标准差为0，所有值相等，直接返回
    for i in range(len(y)):                        # 对每个真实值进行标准化
        y[i] = (y[i] - mean) / std                 # 为增加代码的可读性，此处使用循环

    return X, y                                    # 返回标准化后的数据集和真实值

# 数据预处理数据：对数据进行归一化
# 参数X_matrix：训练或测试数据，维度(m×n)，m行n列，共m个数据，每个数据有n个特征
# 参数y_vector：真实值，维度(m,)，m行1列，共m个真实值
def normalize_data(X_matrix, y_vector=None):
    X = X_matrix.copy()                            # 复制矩阵，防止破坏原始数据集

    m, n = X.shape                                 # 获取数据集的数据量m和特征数n
    for j in range(n):                             # 遍历数据集的每个特征
        max = np.max(X[:, j])                      # 获取该特征（第j列）的最大值
        min = np.min(X[:, j])                      # 获取该特征（第j列）的最小值
        if max == min:                             # 如果max==min，说明所有值都相等
            X[:, j] = 0                            # 将该列所有值都置为0
        else:                                      # 对该特征（第j列）的每个值归一化
            for i in range(m):                     # 为增加代码的可读性，此处使用循环
                X[i][j] = (X[i][j]-min)/(max-min)  # 对第i个数据的第j个特征执行归一化

    if y_vector is None:                           # 若y_vector为空，只对X进行归一化
        return X, None                             # 返回归一化后的数据集

    y = y_vector.copy()                            # 复制列向量，防止破坏原始的真实值
    max = np.max(y)                                # 获取真实值的最大值
    min = np.min(y)                                # 获取真实值的最小值
    if max == min:                                 # 如果max==min，说明所有真实值都相等
        return X, y                                # 对真实值不予处理，直接返回
    else:                                          # 对每一个真实值归一化
        for i in range(len(y)):                    # 为增加代码的可读性，此处使用循环
            y[i] = (y[i]-min) / (max-min)          # 对第i个真实值进行归一化

    return X, y                                    # 返回归一化后的数据集和真实值

# 计算线性回归的均方误差（MSE）
# 参数X：数据矩阵，w：权重向量，b：偏置，y：真实值
def compute_lr_mse(X, w, b, y):
    y_hat = np.dot(X, w) + b                       # 计算预测值
    diff = y_hat - y                               # 计算预测值与真实值之差
    return np.dot(diff, diff) / len(y)             # 计算并返回均方误差

# 计算均方误差（MSE）
# 参数y_hat：预测值，y：真实值
def mean_squared_error(y_hat, y):
    diff = y_hat - y                               # 计算预测值与真实值之差
    return np.dot(diff, diff) / len(y)             # 计算并返回均方误差

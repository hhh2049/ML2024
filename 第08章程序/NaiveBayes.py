# encoding=utf-8
import numpy as np
from scipy.special import logsumexp

class NaiveBayes:  # 朴素贝叶斯模型的实现
    def __init__(self, X, y, alpha=1.0, fit_type="MNB"):
        self.X                = X                               # 训练数据集(m,n)
        self.y                = y                               # 分类标签值(m,)
        self.m, self.n        = X.shape                         # 样本量和特征数
        self.alpha            = alpha                           # 即平滑先验参数λ
        self.fit_type         = fit_type                        # 朴素贝叶斯类型
        # 类别的统计数据，用于BNB、MNB、GNB
        self.K                = len(np.unique(y))               # 获取分类类别数
        self.class_label      = np.zeros(self.K)                # 各类别的标签
        self.class_count      = np.zeros(self.K)                # 各类别的样本数
        self.class_log_prior  = np.zeros(self.K)                # 各类别的对数概率
        # 特征的统计数据，用于BNB、MNB，注意MNB输入的数据是表8-6
        self.feature_count    = np.zeros((self.K, self.n))      # 各类各特征值的数量
        self.feature_log_prob = np.zeros((self.K, self.n))      # 各类各特征值的对数概率
        # 特征的统计数据，用于GNB，注意X_Var仅用于计算epsilon值
        self.mu               = np.zeros((self.K, self.n))      # 各类各特征的均值
        self.sigma            = np.zeros((self.K, self.n))      # 各类各特征的方差
        self.X_var            = np.var(self.X, axis=0)          # 计算数据集各特征的方差
        self.epsilon          = np.max(1e-9 * self.X_var)       # 防止方差为0的一极小值

    def fit(self):  # 拟合数据，训练模型
        y                    = self.y                           # 为缩短下一行代码长度
        unique, count        = np.unique(y, return_counts=True) # 获取各类别和各类样本量
        self.class_label     = unique                           # 保存各类别的标签
        self.class_count     = count                            # 保存各类别的样本量
        self.class_log_prior = np.log(count / len(y))           # 计算各类别的对数先验概率

        if self.fit_type == "BNB" or self.fit_type == "MNB":    # 如果是伯努利或多项式模型
            if self.fit_type == "BNB":                          # 如果是伯努利朴素贝叶斯
                self.X = np.where(self.X >= 1, 1, 0)            # 将>=1的置为1，否则置为0

            for i in range(self.K):                             # 按类别处理数据
                Xk = self.X[np.where(y == self.class_label[i])] # 筛选出第i(k)类的数据
                self.feature_count[i] = np.sum(Xk, axis=0)      # 统计各个特征值的数量
                if self.fit_type == "BNB": Nk = len(Xk)         # 计算该类别有多少个数据
                else: Nk = np.sum(self.feature_count[i])        # 计算该类别有多少个单词

                for j in range(self.n):                         # 对每个特征或每个单词
                    a = self.feature_count[i][j] + self.alpha   # 各特征值或单词的数量
                    if self.fit_type == "BNB":                  # 如果是伯努利朴素贝叶斯
                        b = Nk + 2 * self.alpha                 # 计算分母，0、1两类取值
                    else:                                       # 如果是多项式朴素贝叶斯
                        b = Nk + self.n * self.alpha            # 计算分母，n为词典长度
                    self.feature_log_prob[i][j] = np.log(a/b)   # 计算各特征值的对数概率
        elif self.fit_type == "GNB":                            # 如果高斯朴素贝叶斯
            for i in range(self.K):                             # 按类处理数据
                Xk = self.X[np.where(y == self.class_label[i])] # 筛选出第i(k)类的数据
                self.mu[i] = np.mean(Xk, axis=0)                # 计算第i(k)类数据均值
                self.sigma[i] = np.var(Xk, axis=0)              # 计算第i(k)类数据方差
                self.sigma[i] += self.epsilon                   # 方差加上一个极小值

    def predict_one(self, x):  # 预测一个数据
        log_pk = np.zeros(self.K)                               # 数据属于各类的对数概率

        for k in range(self.K):                                 # 计算数据属于各类的概率
            log_pk[k] = self.class_log_prior[k]                 # 第i类数据的对数概率

            for j in range(self.n):                             # 处理数据x的各个特征的值
                if self.fit_type == "BNB":                      # 如果是伯努利朴素贝叶斯
                    a = self.feature_count[k][j] + self.alpha   # 第k类第j特征1的数量+λ
                    b = self.class_count[k] + 2 * self.alpha    # 第k类样本量+2λ
                    if x[j] == 1: pj = np.log(a / b)            # 计算1的对数概率
                    else: pj = np.log(1 - a / b)                # 计算0的对数概率
                    log_pk[k] += pj                             # 累加各个特征的对数概率
                elif self.fit_type == "MNB":                    # 如果是多项式朴素贝叶斯
                    a = self.feature_count[k][j] + self.alpha   # 第k类第j个单词数量+λ
                    Nk = np.sum(self.feature_count[k])          # 第k类样本的单词数量
                    b = Nk + self.n * self.alpha                # 第k类样本的单词数量+nλ
                    log_pk[k] += x[j] * np.log(a/b)             # 累加各个特征的对数概率
                elif self.fit_type == "GNB":                    # 如果是高斯朴素贝叶斯
                    mu, sigma = self.mu[k][j], self.sigma[k][j] # 第k类第j特征的均值方差
                    a = -np.log(np.sqrt(2 * np.pi * sigma))     # 计算对数高斯概率第1部分
                    b = -(x[j] - mu)**2 / (2 * sigma)           # 计算对数高斯概率第2部分
                    log_pk[k] += a + b                          # 累加各个特征的对数概率

        index  = np.argmax(log_pk)                              # 获取对数概率最大的下标
        label  = self.class_label[index]                        # 确定数据x所属的类别
        lse    = logsumexp(log_pk)                              # 使用logsumexp处理概率
        log_pk = log_pk - lse                                   # x属于各类的对数概率
        p_k    = np.exp(log_pk)                                 # x属于各类的概率

        return label, p_k, log_pk                               # 返回预测结果

    def predict_batch(self, X):  # 预测一个数据集
        if self.fit_type == "BNB":                              # 如果是伯努利朴素贝叶斯
            X = np.where(X >= 1, 1, 0)                          # 将>=1的置为1，否则置为0

        y_hat  = np.zeros(len(X), dtype="int")                  # 存放各数据所属的类
        p_k    = np.zeros((len(X), self.K))                     # 存放各数据属于各类的概率
        log_pk = np.zeros((len(X), self.K))                     # 各数据属于各类的对数概率
        for i in range(len(X)):                                 # 遍历数据集X所有数据
            y_hat[i],p_k[i],log_pk[i] = self.predict_one(X[i])  # 执行预测，保存结果

        return y_hat, p_k, log_pk                               # 返回预测结果

    def predict(self, X):  # 预测各数据属于哪个类
        y_hat = self.predict_batch(X)[0]                        # 调用函数预测
        return y_hat                                            # 返回结果

    def predict_proba(self, X):  # 预测各数据属于各类的概率
        p_k = self.predict_batch(X)[1]                          # 调用函数预测
        return p_k                                              # 返回结果

    def predict_log_proba(self, X):  # 预测各数据属于各类的对数概率
        log_pk = self.predict_batch(X)[2]                       # 调用函数预测
        return log_pk                                           # 返回结果

    def score(self, X, y):  # 计算对数据集X的分类准确率
        if self.fit_type == "BNB":                              # 如果是伯努利朴素贝叶斯
            X = np.where(X >= 1, 1, 0)                          # 将>=1的置为1，否则置为0

        y_hat = self.predict(X)                                 # 预测各个数据所属的类别
        count = np.sum(y_hat == y)                              # 预测值与真实值相等的个数
        score = count / len(y)                                  # 计算分类准确率

        return score                                            # 返回分类准确率

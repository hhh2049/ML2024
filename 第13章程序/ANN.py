# encoding=utf-8
import numpy as np

class ANN:  # 人工神经网络（分类与回归）算法实现
    def __init__(self, X, y, is_classify=True, hidden_layer_sizes=(10, ),
                 activation="logistic", solver='sgd',
                 learning_rate_init=0.001, max_iter=200):
        self.pre_process_dataset(X, y, is_classify)        # 数据预处理
        self.is_classify        = is_classify              # 指明用于分类还是回归
        self.hidden_layer_sizes = hidden_layer_sizes       # 各隐藏层的神经元数量
        self.activation         = activation               # 隐藏层的激活函数类型
        self.solver             = solver                   # 优化算法，只实现SGD
        self.learning_rate      = learning_rate_init       # 学习率
        self.max_iter           = max_iter                 # 最大迭代次数
        self.build_para(hidden_layer_sizes, is_classify)   # 构建和初始化参数

    def pre_process_dataset(self, X, y, is_classify): # 数据预处理
        ones = np.ones(len(X))                             # 生成一个值全为1的向量
        self.X = np.c_[X, ones]                            # 在每个数据后添加一个1

        if is_classify:                                    # 如果用于分类问题
            labels = np.unique(y)                          # 获取分类标签，格式0,1,2...
            self.Y = np.zeros((len(y), len(labels)))       # 定义矩阵Y(m×k)，k为类别数
            for i in range(len(y)):                        # 对真实值向量y进行独热编码
                self.Y[i][y[i]] = 1.0                      # 在相应位置置1(对矩阵Y赋值)
        else:                                              # 如果用于回归问题
            self.Y = y.reshape(-1, 1)                      # 向量y(m,)转为矩阵Y(m×1)

    def build_para(self, hidden_layer_sizes, is_classify): # 构建和初始化参数
        self.W_list     = []                               # 存放各层权重矩阵
        self.grad_list  = []                               # 存放各层权重梯度
        self.in_list    = []                               # 存放各层神经元输入值
        self.out_list   = []                               # 存放各层神经元输出值
        self.delta_list = []                               # 存放各层神经元δ值

        for i in range(len(hidden_layer_sizes)+1):         # 遍历神经网络各层，+1为输出层
            if i == 0:                                     # 如果是第一层
                last = self.X.shape[1]                     # 训练数据的维度(已含偏置项)
            else:                                          # 如果非第一层
                last = hidden_layer_sizes[i-1]             # 上一层神经元的数量
            if i == len(hidden_layer_sizes):               # 如果是输出层
                if is_classify: current = self.Y.shape[1]  # 若是分类，输出层为类别数
                else: current = 1                          # 若是回归，输出层只有1个神经元
            else:                                          # 如果不是输出层（中间隐藏层）
                current = hidden_layer_sizes[i]            # 当前层神经元数量

            W_width = 0.1                                  # W初始值分散度，0<W_width<=1
            W = np.random.randn(current, last) * W_width   # 随机生成当前权重矩阵初始值
            self.W_list.append(W)                          # 将当前权重矩阵加入列表
            grad = np.zeros((current, last))               # 生成梯度矩阵
            self.grad_list.append(grad)                    # 将梯度矩阵加入列表
            self.in_list.append(np.zeros(current))         # 生成当前层u向量并加入列表
            self.out_list.append(np.zeros(current))        # 生成当前层z向量并加入列表
            self.delta_list.append(np.zeros(current))      # 生成当前层δ向量并加入列表

    def f(self, x):  # 计算激活函数值
        if self.activation == "logistic":                  # 若激活函数为逻辑斯蒂函数
            return 1.0 / (1.0 + np.exp(-x))                # 计算和返回逻辑斯蒂函数值
        elif self.activation == "tanh":                    # 若激活函数为双曲正切函数
            return np.tanh(x)                              # 计算和返回双曲正切函数值
        elif self.activation == "relu":                    # 若激活函数为Relu函数
            return np.where(x > 0, x, 0.0)                 # 计算和返回Relu函数值
        else:                                              # 若激活函数为恒等函数y=x
            return x                                       # 计算和返回恒等函数值y=x

    def d(self, x):  # 计算激活函数的一阶导数
        if self.activation == "logistic":                  # 若激活函数为逻辑斯蒂函数
            sigma = 1.0 / (1.0 + np.exp(-x))               # 计算逻辑斯蒂函数值
            return sigma * (1.0 - sigma)                   # 返回逻辑斯蒂函数一阶导数值
        elif self.activation == "tanh":                    # 若激活函数为双曲正切函数
            return 1.0 - np.tanh(x) ** 2.0                 # 返回双曲正切函数一阶导数值
        elif self.activation == "relu":                    # 若激活函数为Relu
            return np.where(x > 0, 1.0, 0.0)               # 返回Relu函数一阶导数值
        else:                                              # 若激活函数为恒等函数y=x
            return 1.0                                     # 返回恒等函数y=x一阶导数值

    def forward_propagation(self, x): # 正向传播
        for i, W in enumerate(self.W_list):                # 遍历神经网络各层
            if i == 0: z = x                               # 若是第一层，z(0)=x
            else: z = self.out_list[i-1]                   # 否则获取上一层的输出值
            self.in_list[i] = np.dot(W, z)                 # 计算当前层的输入值向量
            self.out_list[i] = self.f(self.in_list[i])     # 计算当前层的输出值向量

        if self.is_classify:                               # 若用于分类问题
            z = self.in_list[-1]                           # 获取最后一层的输入值向量
            z = z - np.max(z)                              # 防止溢出，减去最大值
            z = np.exp(z)                                  # 对输入值向量计算指数
            self.out_list[-1] = z / np.sum(z)              # 计算最后一层的输出值向量
            return np.argmax(self.out_list[-1])            # 返回分类预测值向量
        else:                                              # 若用于回归问题
            self.out_list[-1] = self.in_list[-1]           # 输入即输出
            return self.out_list[-1][0]                    # 返回回归预测值(一个标量)

    def backward_propagation(self, x, y):  # 误差反向传播
        layer_count = len(self.W_list)                     # 获取神经网络有多少层
        for i in range(-1, -layer_count-1, -1):            # 从后往前遍历
            if i == -1:                                    # 如果是最后一层
                error = self.out_list[i] - y               # 预测值与真实值之差
                self.delta_list[i] = error                 # 计算当前层δ值
            else:                                          # 如果不是最后一层
                delta = self.delta_list[i+1]               # 为缩短下一行代码长度
                error = np.dot(self.W_list[i+1].T, delta)  # 计算W、δ乘积
                d, in_list_i = self.d, self.in_list[i]     # 为缩短下一行代码长度
                self.delta_list[i] = error * d(in_list_i)  # 计算当前层δ值

            if i == -layer_count:                          # 如果是第一隐藏层
                zin = x                                    # 获取本层输入，即x
            else:                                          # 如果不是第一隐藏层
                zin = self.out_list[i-1]                   # 获取本层输入
            Zin = zin.reshape(-1, 1)                       # 一维数字转换为矩阵
            delta = self.delta_list[i].reshape(-1, 1)      # 一维数字转换为矩阵
            grad = np.dot(delta, Zin.T)                    # 计算梯度
            self.grad_list[i] = grad                       # 保存梯度

    def update_model_parameters(self):  # 更新模型参数
        layer_count = len(self.W_list)                     # 获取神经网络有多少层
        for i in range(-1, -layer_count-1, -1):            # 从后往前遍历
            grad = self.grad_list[i]                       # 获取当前层的权重梯度
            self.W_list[i] -= self.learning_rate * grad    # 更新当前层的权重梯度

    def fit(self):  # 拟合数据，训练模型
        X, Y = self.X, self.Y                              # 为缩短代码长度
        for _ in range(self.max_iter):                     # 遍历训练轮数
            for i in range(len(X)):                        # 遍历训练数据
                self.forward_propagation(X[i])             # 前向传播
                self.backward_propagation(X[i], Y[i])      # 误差反向传播
                self.update_model_parameters()             # 更新各层权重参数

    def predict_one(self, x):  # 预测一个数据
        return self.forward_propagation(x)                 # 使用前向传播预测一个数据

    def predict(self, X):  # 预测一个数据集
        ones = np.ones(len(X))                             # 生成一个值全为1的向量
        X = np.c_[X, ones]                                 # 在每个数据最后添加一个1

        y_hat = np.zeros(len(X))                           # 定义预测值向量
        for i in range(len(X)):                            # 遍历每个数据
            y_hat[i] = self.predict_one(X[i])              # 预测一个数据

        return y_hat                                       # 返回预测值向量

    def score(self, X, y):  # 计算分类或回归得分
        y_hat = self.predict(X)                            # 获取数据集X的预测值

        if self.is_classify:                               # 如果是分类问题
            count = np.sum(y_hat == y)                     # 预测值与真实值相等的个数
            return count / len(y)                          # 计算并返回分类得分
        else:                                              # 如果是回归问题
            diff = y - y_hat                               # 计算真实值与预测值之差
            mse = np.dot(diff, diff) / len(X)              # 计算MSE
            y_mean = np.mean(y)                            # 计算真实值的平均值
            diff = y - y_mean                              # 计算真实值与平均值之差
            var = np.dot(diff, diff) / len(X)              # 计算VAR
            return 1.0 - mse / var                         # 计算并返回回归得分

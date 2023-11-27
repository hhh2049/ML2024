# encoding=utf-8
import numpy as np
from sklearn import datasets
from GMM_EM import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture as GM

def main():
    # 测试数据一：鸢尾花数据集
    iris_data = datasets.load_iris()
    X, y = iris_data.data, iris_data.target
    m, n = X.shape
    # 使用自实现的高斯混合模型
    our_gmm = GaussianMixture(X, y, n_components=3)
    our_gmm.train()
    y_predict = our_gmm.predict(X)
    our_gmm.print_values()
    score = adjusted_rand_score(y, y_predict)      # 计算调整兰德指数，下一章介绍
    print("ARI score:", score)                     # 打印调整兰德指数
    # 使用官方库的高斯混合模型
    w, Mu = np.ones(3)/3, X[:3]                    # 初始权重相等，均值使用前K个数据
    Sigma = np.zeros((3, n, n))                    # 用于存放3个高斯分布的协方差
    for i in range(3):                             # 这里的3为分类数，也是高斯分布数
        Sigma[i] = np.eye(n)                       # 协方差初始化为单位矩阵
    skl_gmm = GM(n_components=3, weights_init=w, means_init=Mu, precisions_init=Sigma)
    skl_gmm.fit(X, y)
    print("\nScikit-learn result:")
    print("w: " + str(skl_gmm.weights_))
    print("Mu: \n" + str(skl_gmm.means_))
    y_predict = skl_gmm.predict(X)
    score = adjusted_rand_score(y, y_predict)
    print("ARI score:", score, "\n")
    # exit()

    # 测试数据二：自动生成数据
    # 设置3个高斯分布参数：权重，均值、协方差
    w = np.array([0.3, 0.6, 0.1])
    means = np.array([
        [2.5, 8],
        [8, 2.5],
        [10, 10]
    ])
    covs = np.array([
        [[2, 1], [1, 2]],
        [[2, 1], [1, 2]],
        [[2, 0], [0, 2]]
    ])
    # 随机生成1000个数据，按比例服从3个高斯分布
    X = np.zeros((1000, 2))
    X[0: 300]    = np.random.multivariate_normal(means[0], covs[0], 300)
    X[300: 900]  = np.random.multivariate_normal(means[1], covs[1], 600)
    X[900: 1000] = np.random.multivariate_normal(means[2], covs[2], 100)
    m, n = X.shape
    # 打乱1000个数据的次序
    index_random = np.arange(1000)
    np.random.shuffle(index_random)
    X = X[index_random]
    # 使用自实现的高斯混合模型
    our_gmm = GaussianMixture(X, None, n_components=3)
    our_gmm.train()
    our_gmm.print_values()
    # 使用官方库的高斯混合模型
    w, Mu = np.ones(3)/3, X[:3]
    Sigma = np.zeros((3, n, n))
    for i in range(3):
        Sigma[i] = np.eye(n)
    skl_gmm = GM(n_components=3, weights_init=w, means_init=Mu, precisions_init=Sigma)
    skl_gmm.fit(X, None)
    print("\nScikit-learn result:")
    print("w: " + str(skl_gmm.weights_))
    print("Mu: \n" + str(skl_gmm.means_))

if __name__ == "__main__":
    main()

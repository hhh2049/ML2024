# encoding=utf-8
from SVM_SMO import SVM_SMO
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split

def main():
    # 测试自实现和官方库的支持向量机模型
    iris_data = datasets.load_iris()                                # 导入鸢尾花数据集
    X, y = iris_data.data[50:150], iris_data.target[50:150]         # 第二三类线性不可分
    # X, y = iris_data.data[0:100], iris_data.target[0:100]         # 第一二类线性可分
    y[0:50], y[50:100] = 1, -1                                      # 分类标签设为1和-1

    Z = train_test_split(X, y, test_size=0.25, random_state=0)      # 划分训练集和测试集
    (X_train, X_test, y_train, y_test) = Z                          # 训练集和测试集赋值

    our_svm = SVM_SMO(X_train, y_train)                             # 定义线性核SVM对象
    # our_svm = SVM_SMO(X_train, y_train, kernel="rbf", gamma=0.1)  # 定义高斯核SVM对象
    our_svm.train()                                                 # 训练模型
    our_train_score = our_svm.score(X_train, y_train)               # 计算训练得分
    our_test_score  = our_svm.score(X_test, y_test)                 # 计算测试得分
    print("\nour own svm train score = %.6f" % our_train_score)     # 打印训练得分
    print("our own svm test  score = %.6f" % our_test_score)        # 打印测试得分

    skl_svm = svm.SVC(kernel="linear")                              # 定义线性核SVM对象
    # skl_svm = svm.SVC(kernel="rbf", gamma=0.1)                    # 定义高斯核SVM对象
    skl_svm.fit(X_train, y_train)                                   # 训练模型
    skl_train_score = skl_svm.score(X_train, y_train)               # 计算训练得分
    skl_test_score = skl_svm.score(X_test, y_test)                  # 计算测试得分
    print("sklearn svm train score = %.6f" % skl_train_score)       # 打印训练得分
    print("sklearn svm test  score = %.6f" % skl_test_score)        # 打印测试得分

if __name__ == "__main__":
    main()

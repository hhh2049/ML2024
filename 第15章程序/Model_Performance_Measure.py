# encoding=utf-8
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, classification_report

# 分类问题的性能度量
y_true    = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
y_predict = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
# 准确率（正确率）
score = accuracy_score(y_true, y_predict)
print("accuracy_score: ", score)
# 查准率
score = precision_score(y_true, y_predict, pos_label=1)
print("precision_score:", score)
# 查全率
score = recall_score(y_true, y_predict, pos_label=1)
print("recall_score:   ", score)
# F1值
score = f1_score(y_true, y_predict, pos_label=1)
print("f1_score:       ", score)
# Fβ值
score = fbeta_score(y_true, y_predict, beta=1.0, pos_label=1)
print("fbeta_score:    ", score)
# 分类预测的混淆矩阵
result = confusion_matrix(y_true, y_predict)
print("confusion_matrix:\n", result)
# 分类预测的主要性能指标
result = classification_report(y_true, y_predict)
print("classification_report:\n", result)

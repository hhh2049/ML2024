# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

y_true   = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
print("recall:    ", recall[::-1])      # 逆序输出查全率数组
print("precision: ", precision[::-1])   # 逆序输出查准率数组
print("thresholds:", thresholds[::-1])  # 逆序输出阈值数组

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(recall, precision, label="PR")
ax.set_title("P-R")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.legend(loc="best")
ax.grid()
plt.show()

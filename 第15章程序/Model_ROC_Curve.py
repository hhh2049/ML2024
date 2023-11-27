# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

y_true   = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
print("auc:", roc_auc_score(y_true, y_scores))
print("fpr:", fpr)
print("tpr:", tpr)
print("threshold:", thresholds)

fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
ax.plot(fpr, tpr, label="ROC")
ax.set_title("ROC")
ax.set_xlabel("fpr")
ax.set_ylabel("tpr")
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.legend(loc="best")
ax.grid()
plt.show()
# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC

X, y        = load_digits(return_X_y=True)
param_range = np.logspace(-6, -1, 5)
z = validation_curve(SVC(), X, y,
                     param_name="gamma", param_range=param_range,
                     cv=5, scoring="accuracy")
(train_scores, test_scores) = z
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std  = np.std(train_scores,  axis=1)
test_scores_mean  = np.mean(test_scores,  axis=1)
test_scores_std   = np.std(test_scores,   axis=1)

fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
ax.semilogx(param_range, train_scores_mean, "-", label="Training Accuracy", color="r")
ax.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2, color="r")
ax.semilogx(param_range, test_scores_mean, "--", label="Test Accuracy", color="g")
ax.fill_between(param_range, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.2, color="g")
ax.set_title("Validation Curve")
ax.set_xlabel("gamma")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.legend(loc="best")
ax.grid()
plt.show()



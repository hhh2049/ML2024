# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

X, y      = load_digits(return_X_y=True)
sizes     = np.linspace(0.1, 1.0, 5)
cv        = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = GaussianNB()
# cv        = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# estimator = SVC(gamma=0.001)
ret = learning_curve(estimator, X, y, cv=cv, scoring="accuracy", train_sizes=sizes)
(train_sizes, train_score, test_score) = ret
train_score_mean = np.mean(train_score, axis=1)
train_score_std  = np.std(train_score,  axis=1)
test_score_mean  = np.mean(test_score,  axis=1)
test_score_std   = np.std(test_score,   axis=1)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.plot(train_sizes, train_score_mean, 'o-', label="Training Score", color="r")
ax.fill_between(train_sizes, train_score_mean - train_score_std,
                train_score_mean + train_score_std, alpha=0.2, color="r")
ax.plot(train_sizes, test_score_mean, 'o--', label="Test Score", color="g")
ax.fill_between(train_sizes, test_score_mean - test_score_std,
                test_score_mean + test_score_std, alpha=0.2, color="g")
ax.set_title("Learning Curve(Naive Bayes)")
# ax.set_title("Learning Curve(SVM RBF Kernel gamma=0.001)")
ax.set_xlabel("Training Examples")
ax.set_ylabel("Score")
ax.set_ylim(0.7, 1.01)
ax.legend(loc="best")
ax.grid()
plt.show()

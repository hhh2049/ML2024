# encoding=utf-8
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from scipy.stats import uniform

X, y = load_iris(return_X_y=True)
estimator = LogisticRegression(solver='saga', tol=1e-2, max_iter=200, random_state=0)
distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
clf = RandomizedSearchCV(estimator, distributions, random_state=0)
clf.fit(X, y)
print("Best score:%0.6f" % clf.best_score_)
print("Best parameters:", clf.best_params_, "\n")
print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds  = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%f (+/-%0.04f) for %s" % (mean, std * 2, params))
print("\nDetailed classification report:")
y_true, y_predict = y, clf.predict(X)
print(classification_report(y_true, y_predict))
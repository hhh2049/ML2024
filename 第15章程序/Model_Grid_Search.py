# encoding=utf-8
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.svm import SVC

X, y = datasets.load_digits(return_X_y=True)
z = train_test_split(X, y, test_size=0.25, random_state=0)
(X_train, X_test, y_train, y_test) = z
tuned_parameters = [{'kernel': ['rbf'],
                     'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'],
                     'C': [1, 10, 100, 1000]}]
clf = GridSearchCV(SVC(), tuned_parameters)
clf.fit(X_train, y_train)
print("Best score:%0.6f" % clf.best_score_)
print("Best parameters:", clf.best_params_, "\n")

print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds  = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%f (+/-%0.04f) for %s" % (mean, std * 2, params))
print("\nDetailed classification report:")
y_true, y_predict = y_test, clf.predict(X_test)
print(classification_report(y_true, y_predict))

from sklearn import svm
from sklearn.svm import OneClassSVM

X = [[0, 0], [1, 1]]
y = [0, 1]
# clf = svm.SVC()
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.5)
clf.fit(X, y)
print clf.decision_function([[2., 2.]])

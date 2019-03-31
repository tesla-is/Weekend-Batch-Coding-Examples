import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()

from sklearn.datasets import load_breast_cancer
dataset1 = load_breast_cancer()

X = dataset.data
y = dataset.target

plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], c = "r", label = "Setosa")
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], c = "g", label = "Versicolor")
plt.scatter(X[:, 0][y == 2], X[:, 1][y == 2], c = "b", label = "Verginica")
plt.legend()
plt.show()

plt.scatter(X[:, 2][y == 0], X[:, 3][y == 0], c = "r", label = "Setosa")
plt.scatter(X[:, 2][y == 1], X[:, 3][y == 1], c = "g", label = "Versicolor")
plt.scatter(X[:, 2][y == 2], X[:, 3][y == 2], c = "b", label = "Verginica")
plt.legend()
plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_pred1 = knn.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm1 = confusion_matrix(y_test, y_pred1)

log_reg.score(X_train, y_train)
log_reg.score(X_test, y_test)
log_reg.score(X, y)

knn.score(X_train, y_train)
knn.score(X_test, y_test)
knn.score(X, y)







































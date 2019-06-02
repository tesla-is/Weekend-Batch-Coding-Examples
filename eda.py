import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('dataset/housing.csv')
dataset.info()



corr_mat = dataset.corr()
sns.heatmap(corr_mat, annot = True)

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target



from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

log_reg.score(X, y)

y_pred = log_reg.predict(X)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y, y_pred)
recall_score(y, y_pred)
f1_score(y, y_pred)

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

log_reg.score(X, y)

y_pred = log_reg.predict(X)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y, y_pred, average = 'macro')
recall_score(y, y_pred, average = 'macro')
f1_score(y, y_pred, average = 'macro')


























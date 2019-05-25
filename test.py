import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 2)
dtf.fit(X, y)


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X, y)


from sklearn.ensemble import VotingClassifier
vot = VotingClassifier([('log', log_reg), ('DT', dtf), ('NB', nb)])
vot.fit(X, y)

from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(log_reg)
bag.fit(X, y)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X, y)


dtf.score(X, y)
log_reg.score(X, y)
nb.score(X, y)
vot.score(X, y)
bag.score(X, y)
rf.score(X, y)


























param_grid = [{'criterion' : ['gini', 'entropy']},
               {'max_depth' : [3, 4, 5, 6, 7, 8, 9]}]


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(dtf, param_grid)
grid.fit(X, y)

grid.best_estimator_
grid.best_params_
grid.best_score_















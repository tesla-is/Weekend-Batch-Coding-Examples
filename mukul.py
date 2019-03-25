import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X = 2 * np.random.randn(100, 1)
y = 7 + 5 * X + np.random.randn(100, 1)

plt.scatter(X, y)
plt.show()

X_c = np.c_[np.ones(100), X]

theta = np.linalg.inv(X_c.T @ X_c) @ (X_c.T @ y)

mat = np.array([[1, 2], [3, 4]])
mat * mat
mat @ mat

dataset = pd.read_excel('dataset/blood.xlsx')
X = dataset.iloc[2:, 1].values
y = dataset.iloc[2:, -1].values
X = X.reshape(-1, 1)

plt.scatter(X, y)
plt.show()

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

plt.scatter(X, y)
plt.plot(X, lin_reg.predict(X), c = "r")
plt.show()

lin_reg.predict([[21]])
lin_reg.predict([[31]])
lin_reg.predict([[41]])

lin_reg.score(X, y)






















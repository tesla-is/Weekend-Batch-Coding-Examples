import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

m = 100
X = 6 * np.random.randn(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

plt.scatter(X, y)
plt.axis([-3, 3, 0, 10])
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2, include_bias = False)
X_poly = poly.fit_transform(X)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

X_new = np.linspace(-3, 3, 100).reshape(m, 1)
X_new_poly = poly.fit_transform(X_new)
y_new = lin_reg.predict(X_new_poly)


plt.scatter(X, y)
plt.plot(X_new, y_new, c = "r")
plt.axis([-3, 3, 0, 10])
plt.show()

from sklearn.svm import SVC
svm = SVC()







































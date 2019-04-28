import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples = 300, centers = 5, cluster_std = 0.6)

plt.scatter(x[:, 0], x[:, 1])
plt.show()

from sklearn.cluster import KMeans

wcv = []

for i in range(1, 11):
    km = KMeans(n_clusters = i)
    km.fit(x)
    wcv.append(km.inertia_)

plt.plot(range(1, 11), wcv)
plt.show()

km = KMeans(n_clusters = 5)
y_pred = km.fit_predict(x)


plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], c = "r", label = "Smart Customers")
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], c = "b", label = "Target Customers")
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], c = "g", label = "Smart Customers")
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], c = "y", label = "Target Customers")
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], c = "magenta", label = "Smart Customers")
plt.legend()
plt.show()


















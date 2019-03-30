import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.arange(-10, 10, 0.01)
y = 1 / (1 + np.power(np.e, -x))
y1 = np.power(np.e, -x) / (1 + np.power(np.e, -x))

plt.plot(x, y)
plt.show()

plt.plot(x, y1)
plt.show()

line = 0.5 * x + 3

plt.plot(x, line)
plt.show()

sig = 1 / (1 + np.power(np.e, -line))

plt.plot(x, sig)
plt.show()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





















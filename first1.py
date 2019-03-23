import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

my_list = list(range(1000000))
my_arr = np.array(range(1000000))

%time for i in range(10): my_list2 = my_list * 2
%time for i in range(10): my_arr2 = my_arr * 2

plt.scatter([1, 2, 3], [4, 5, 6])
plt.show()

plt.plot([1, 2, 3], [4, 5, 6])
plt.show()

a = pd.Series([1, 2, 3, 4, 5], ['a', 'b', 'c', 'd', 'e'])

b = pd.DataFrame({1 : [1, 2, 3, 4, 5],
                  2 : [1, 2, 3, 5, 5]})

c = pd.DataFrame([[1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5]])


dataset = pd.read_csv('dataset/housing.csv') 

plt.scatter(dataset['total_bedrooms'], dataset['total_rooms'])
plt.show()

pd.scatter_matrix(dataset)

dataset = pd.read_csv('dataset/Data_Pre.csv')
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imp.fit(X[:, 0:2])
X[:, 0:2] = imp.transform(X[:, 0:2])

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:, 2] = lab.fit_transform(X[:, 2])
y = lab.fit_transform(y)
lab.classes_

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [2])
X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

dataset = pd.read_csv('dataset/sal.csv', names = ['age',
                                                  'state',
                                                  'country'])









































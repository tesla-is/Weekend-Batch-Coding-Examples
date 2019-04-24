import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('dataset/StudentsPerformance.csv')

dataset.head()
dataset.tail()
dataset.sample(5)
dataset.info()
dataset.describe()
dataset.dtypes
dataset.corr()
dataset.isnull().values.any()
dataset.isnull().sum()

for i, col in enumerate(dataset.columns):
    print(i+1, " . column is ", col)

dataset['gender'].value_counts()
dataset['gender'].unique()

sns.set(style = 'whitegrid')
sns.barplot(x = dataset['gender'].value_counts().index, y = dataset['gender'].value_counts(), palette = 'Blues_d',
            hue = ['female', 'male'])
plt.legend(loc = 8)
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Bar Plot')
plt.show()

plt.figure(figsize = (7, 7))
sns.barplot(x = dataset['race/ethnicity'].value_counts().index, 
            y = dataset['race/ethnicity'].value_counts().values)


sns.barplot(x = dataset['parental level of education'], y = dataset['writing score'], hue = dataset['gender'],
            data = dataset)














































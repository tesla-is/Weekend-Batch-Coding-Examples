import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('dataset/StudentsPerformance.csv')

dataset.info()
dataset.describe()
dataset.isnull().sum()

dataset.columns = ['gender', 'race', 'ped', 'lunch', 'test', 'math', 'reading', 'writing'] 

plt.hist(dataset['math'], bins = 100)
plt.show()

plt.hist(dataset['reading'], bins = 100)
plt.show()

plt.hist(dataset['writing'], bins = 100)
plt.show()

pd.scatter_matrix(dataset)
sns.pairplot(dataset)

sns.barplot(dataset['gender'].value_counts().index, dataset['gender'].value_counts(), hue = ['female', 'male'])

sns.barplot(dataset['gender'], dataset['math'], hue = dataset['gender'])
sns.barplot(dataset['gender'], dataset['reading'], hue = dataset['gender'])
sns.barplot(dataset['gender'], dataset['writing'], hue = dataset['gender'])

sns.barplot(dataset['race'], dataset['math'], hue = dataset['gender'])
sns.barplot(dataset['race'], dataset['reading'], hue = dataset['gender'])
sns.barplot(dataset['race'], dataset['writing'], hue = dataset['gender'])

sns.barplot(dataset['ped'], dataset['math'], hue = dataset['gender'])
sns.barplot(dataset['ped'], dataset['reading'], hue = dataset['gender'])
sns.barplot(dataset['ped'], dataset['writing'], hue = dataset['gender'])

sns.barplot(dataset['lunch'], dataset['math'], hue = dataset['gender'])
sns.barplot(dataset['lunch'], dataset['reading'], hue = dataset['gender'])
sns.barplot(dataset['lunch'], dataset['writing'], hue = dataset['gender'])

sns.barplot(dataset['test'], dataset['math'], hue = dataset['gender'])
sns.barplot(dataset['test'], dataset['reading'], hue = dataset['gender'])
sns.barplot(dataset['test'], dataset['writing'], hue = dataset['gender'])

sns.boxplot(dataset['math'])
sns.boxplot(dataset['reading'])
sns.boxplot(dataset['writing'])

sns.boxplot(dataset['gender'], dataset['math'])
sns.boxplot(dataset['gender'], dataset['reading'])
sns.boxplot(dataset['gender'], dataset['writing'])

from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

plt.imshow(train_images[15000])

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (28 * 28,)))
network.add(layers.Dense(10, activation = 'softmax'))

train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_images = train_images / 255
test_images = test_images / 255

from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

network.fit(train_images, train_labels, epochs = 5, batch_size = 128)
test_loss, test_accuracy = network.evaluate(test_images, test_labels)

print(test_accuracy)
print(test_loss)















































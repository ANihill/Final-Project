# Adam Nihill, 28/04/2019
# Investigation into Fisher's Iris Dataset

# Import dataset from scikit
from sklearn.datasets import load_iris

# Import required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Created a variable named data that contains the iris dataset
data = load_iris().data

#print(data)

# Print the shape of array
print(data.shape)

# Extracted class labels and combined them with the dataset using numpy
labels = load_iris().target
labels = np.reshape(labels,(150,1))

data = np.concatenate([data,labels],axis=-1)

# Used Pandas to add a column with attribute names
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
dataset = pd.DataFrame(data,columns=names)

# Replace numbers values representing species with species name
dataset['species'].replace(0, 'Iris-setosa',inplace=True)
dataset['species'].replace(1, 'Iris-versicolor',inplace=True)
dataset['species'].replace(2, 'Iris-virginica',inplace=True)

#print(dataset.head())

print(dataset.groupby('species').size())

# Used Pandas describe Module to summarise the data
print(dataset.describe())

# Generated a Scatterplot matrix
# http://benalexkeen.com/scatter-matrices-using-pandas/
pd.scatter_matrix(dataset)
plt.show()

# Histogram
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
fig = plt.figure(figsize = (8,8))
ax = fig.gca()
dataset.hist(ax=ax)
plt.show()

# Scatterplot comparing sepal length and width
plt.figure(4, figsize=(10, 8))

plt.scatter(data[:50, 0], data[:50, 1], c='r', label='Iris-setosa')

plt.scatter(data[50:100, 0], data[50:100, 1], c='g',label='Iris-versicolor')

plt.scatter(data[100:, 0], data[100:, 1], c='b',label='Iris-virginica')

plt.xlabel('Sepal length',fontsize=20)
plt.ylabel('Sepal width',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Sepal length vs. Sepal width',fontsize=20)
plt.legend(prop={'size': 18})
plt.show()

# Scatterplot comparing petal length and width
plt.figure(4, figsize=(8, 8))

plt.scatter(data[:50, 2], data[:50, 3], c='r', label='Iris-setosa')

plt.scatter(data[50:100, 2], data[50:100, 3], c='g',label='Iris-versicolor')

plt.scatter(data[100:, 2], data[100:, 3], c='b',label='Iris-virginica')
plt.xlabel('Petal length',fontsize=15)
plt.ylabel('Petal width',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Petal length vs. Petal width',fontsize=15)
plt.legend(prop={'size': 20})
plt.show()

# Box and Whisker Plot
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()
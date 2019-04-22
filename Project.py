# import dataset from scikit
from sklearn.datasets import load_iris

# import libraries
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

# Used Pandas describe Module to summarise the data
print(dataset.describe())

print(dataset.groupby('species').size())
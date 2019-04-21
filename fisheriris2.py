# Exploration of data following online reference
# https://www.kaggle.com/lalitharajesh/iris-dataset-exploratory-data-analysis#
# import load_iris function from datasets module
from sklearn.datasets import load_iris

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
type(iris) 

#print the iris dataset
# Each row represents the flowers and each column represents the length and width.
print (iris.data)
iris.data.shape

# print the names of the four features
print (iris.feature_names)

# print the integers representing the species of each observation
print (iris.target)

# print the encoding scheme for species; 0 = Setosa , 1=Versicolor, 2= virginica
print (iris.target_names)

# Check the types of the features and response
type('iris.data')
type('iris.target')

# Check the shape of the features 
#(first dimension = (ROWS) ie number of observations, second dimensions = (COLUMNS) ie number of features)
iris.data.shape

# Extract the values for features and create a list called featuresAll
featuresAll=[]
features = iris.data[: , [0,1,2,3]]
features.shape

# Extract the values for targets
targets = iris.target
targets.reshape(targets.shape[0],-1)
targets.shape

# Every observation gets appended into the list once it is read. For loop is used for iteration process
for observation in features:
    featuresAll.append([observation[0] + observation[1] + observation[2] + observation[3]])
print (featuresAll)

# Plotting the Scatter plot
plt.scatter(featuresAll, targets, color='red', alpha =1.0)
plt.rcParams['figure.figsize'] = [10,8]
plt.title('Iris Dataset scatter Plot')
plt.xlabel('Features')
plt.ylabel('Targets')
plt.show()

#Finding the relationship between Sepal Length and Sepal width
featuresAll = []
targets = []
for feature in features:
    featuresAll.append(feature[0]) #Sepal length
    targets.append(feature[1]) #sepal width

groups = ('Iris-setosa','Iris-versicolor','Iris-virginica')
colors = ('blue', 'green','red')
data = ((featuresAll[:50], targets[:50]), (featuresAll[50:100], targets[50:100]), 
        (featuresAll[100:150], targets[100:150]))

for item, color, group in zip(data,colors,groups): 
    #item = (featuresAll[:50], targets[:50]), (featuresAll[50:100], targets[50:100]), (featuresAll[100:150], targets[100:150])
    x, y = item
    plt.scatter(x, y,color=color,alpha=1)
    plt.title('Iris Dataset scatter Plot')
plt.xlabel('sepal length')
plt.ylabel('Sepal width')
plt.show()

#Finding the relationship between Petal Length and Petal width
featuresAll = []
targets = []
for feature in features:
    featuresAll.append(feature[2]) #Petal length
    targets.append(feature[3]) #Petal width

groups = ('Iris-setosa','Iris-versicolor','Iris-virginica')
colors = ('blue', 'green','red')
data = ((featuresAll[:50], targets[:50]), (featuresAll[50:100], targets[50:100]), 
        (featuresAll[100:150], targets[100:150]))

for item, color, group in zip(data,colors,groups): 
    #item = (featuresAll[:50], targets[:50]), (featuresAll[50:100], targets[50:100]), (featuresAll[100:150], targets[100:150])
    x0, y0 = item
    plt.scatter(x0, y0,color=color,alpha=1)
    plt.title('Iris Dataset scatter Plot')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()

# Initial exploration of data using online reference 
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# Load Libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# Load Dataset
url = "/Users/adamnihill/Documents/Python/Final_Project/Iris.csv"
dataset = pandas.read_csv(url)

# shape
print(dataset.shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()
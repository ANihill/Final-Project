# Programme and Scripting Final Project

This repository contains an ivestigation into the Fisher Iris dataset as part of the final project for the module Programming and Scripting at GMIT.
[See here for instructions.](https://github.com/ianmcloughlin/project-pands/raw/master/project.pdf)

## Ensure that python is installed in order to run code

## Download
[Download Link](https://github.com/ANihill/Final-Project)

## Background information about the data set

### Iris Dataset 
- https://archive.ics.uci.edu/ml/datasets/iris
  - Best known database to be found in pattern recognition literature
  - 3 classes of 50 instances each
  - Each class refers to a type of iris plant
  - One class is linearly separable for the other 2
- https://en.wikipedia.org/wiki/Iris_flower_data_set
  - Quantify the morphologic variation of Iris flowers of three related species
  - Four features measured: length and width of sepals and petal, in centimetres
  - Fisher developed a linear discriminant model to distinguish the species from each other
- https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1469-1809.1936.tb02137.x
  - Published by Ronald Fisher in his 1936 paper *The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis.*

## Summary of investigations 
Investigation into the data began with research on Fisher's dataset itself. This was done to better understand the data that it contained. Fisher's dataset contained information on three Iris species, Iris Setosa, Iris Versicolor, and Iris Virginica. Four features, petal length and width, and sepal length and width, were measured on fifty samples of each species. Fisher determined that one of the species was linearly separable from the other two.

I then researched python code that could be applied to dataset, as seen in the references below. I first ran this code through the iPython interpreter to become familiar with its output. I determined that the libraries required to do this were Pandas, NumPy, and MatPlotLib, as well as SciKit, which was used to load that dataset. Some manipulation of the data was required in order to improve legibility. This involved extracting the class labels and combining them with the dataset. Then Pandas was used to add a column with the attibute names, petal length, petal width, sepal length and sepal width, as well as replacing the number values with the species with the species. 

The next step was to run some basic code to get an overall feel for the data. I determined the shape of the data, (150,4), and ran ``(dataset.describe())`` to calculate the mean, standard deviation, min, and max of the dataset.

|      | sepal-length | sepal-width | petal-length | petal-width |
| ---- | ------------ | ----------- | ------------ | ----------- |
| count|  150.000000  | 150.000000  |  150.000000  | 150.000000  |
| mean |    5.843333  |   3.057333  |    3.758000  |   1.199333  |
| std  |    0.828066  |   0.435866  |    1.765298  |   0.762238  |
| min  |    4.300000  |   2.000000  |    1.000000  |   0.100000  |
| max  |    7.900000  |   4.400000  |    6.900000  |   2.500000  |


From here it was import to visualise the data. A scatterplot matrix was compiled in order to isolate relationships that supported Fisher's assertion that one of the species was linearly separable from the others.

![Alt Text](https://github.com/ANihill/Final-Project/blob/master/Figure_5.png "Scatterplot Matrix")

Two relationships demonstrated a strong correlation to Fishers statement; Petal-Width and Petal-Length, and more obviously,
Sepal-Width and Sepal-Length.

![Alt Text](https://github.com/ANihill/Final-Project/blob/master/Figure_2.png "Petal-Length vs. Petal-Width")

![Alt Text](https://github.com/ANihill/Final-Project/blob/master/Figure_1.png "Sepal-Length vs. Sepal-Width")

The Sepal-Length vs. Sepal- width scatterplot clearly demonstrates that the Iris Setosa is linearly separable from the other two species.

## List of references used in completing the project 
- https://archive.ics.uci.edu/ml/datasets/iris
- https://en.wikipedia.org/wiki/Iris_flower_data_set
- https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1469-1809.1936.tb02137.x
- https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
- https://www.kaggle.com/lalitharajesh/iris-dataset-exploratory-data-analysis/log
- https://www.datacamp.com/community/tutorials/introduction-machine-learning-python
- http://benalexkeen.com/scatter-matrices-using-pandas/
- https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html

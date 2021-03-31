# An analysis of Fisher’s Iris data set
# Author: Andrew Scott

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading in Fisher’s Iris data set from a .data file
# As file did not include column names, they were added using 'names' as an argument
iris = pd.read_csv('iris.data', names = ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class"])


# Overall summary 
print(iris.describe())

# Summaries seperated by class
# Summary of sepal length


# Summary of sepal width


# Summary of petal length


# Summary of petal width


# Histogram of of sepal length


# Histogram of sepal width


# Histogram of petal length


# Histogram of petal width


# Scatterplots

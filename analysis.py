# An analysis of Fisher’s Iris data set
# Author: Andrew Scott

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading in Fisher’s Iris data set from a .data file
# As file did not include column names, they were added using 'names' as an argument
iris = pd.read_csv('iris.data', names = ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class"])


# Overall summary 
# Info to add dataframe to txt file found at: https://stackoverflow.com/questions/31247198/python-pandas-write-content-of-dataframe-into-text-file
# Seperator set to comma so txt file data can easily be added to an excel sheet as a csv file if needed 
# "r" before the filename means the file path is read literally by python, to avoid conflict with any characters 
# that may have other purposes in python https://discuss.python.org/t/what-does-r-define-in-file-path/3646
# First dataframe to txt file output set to write, while rest are set to append. This is to avoid tables being repeated each time 
# this code is run
iris.describe()
iris.describe().to_csv(r'iris_summary.txt', sep=',', mode='w')


# Summaries seperated by class
# Summary of sepal length
iris[["sepal length in cm", "class"]].groupby("class").describe().to_csv(r'iris_summary.txt', index="class", sep=',', mode='a')

# Summary of sepal width
iris[["sepal width in cm", "class"]].groupby("class").describe().to_csv(r'iris_summary.txt', index="class", sep=',', mode='a')

# Summary of petal length
iris[["petal length in cm", "class"]].groupby("class").describe().to_csv(r'iris_summary.txt', index="class", sep=',', mode='a')

# Summary of petal width
iris[["petal width in cm", "class"]].groupby("class").describe().to_csv(r'iris_summary.txt', index="class", sep=',', mode='a')


# Histogram of of sepal length


# Histogram of sepal width


# Histogram of petal length


# Histogram of petal width


# Scatterplots

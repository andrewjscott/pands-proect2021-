# An analysis of Fisher’s Iris data set
# Author: Andrew Scott

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Loading in Fisher’s Iris data set from a .data file
# As file did not include column names, they were added using 'names' as an argument
iris = pd.read_csv('iris.data', names = ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class"])

# Add check to see if data is complete, no missing values etc


# Overall summary
# Checks for a text file by the specified name, and if it does not exist that file is created. It is in write mode meaning anything written to the 
# file will be 
with open("iris_summary.txt", "w") as f:
    # Creates a pandas dataframe containing basic descriptive statistics from the dataset. This is then converted to a string so it can be 
    # written to the text file. Two new lines are also specidied to seperate this output from subsequent outputs
    descriptives = iris.describe()
    f.write(descriptives.to_string() + "\n\n") 
    
    sepal_length_summary = iris[["sepal length in cm", "class"]].groupby("class").describe()
    f.write(sepal_length_summary.to_string() + "\n\n")

    sepal_width_summary = iris[["sepal width in cm", "class"]].groupby("class").describe()
    f.write(sepal_width_summary.to_string() + "\n\n")

    petal_length_summary = iris[["petal length in cm", "class"]].groupby("class").describe()
    f.write(petal_length_summary.to_string() + "\n\n")

    petal_width_summary = iris[["petal width in cm", "class"]].groupby("class").describe()
    f.write(petal_width_summary.to_string() + "\n\n")


# Also decided to output the summaries to csv files as these can be easier to work with should someone want to do further analysis using 
# the output summaries
# Info to add dataframe to csv file found at: https://stackoverflow.com/questions/31247198/python-pandas-write-content-of-dataframe-into-text-file
# Seperator set to comma so csv file data can easily be added to an excel sheet if needed 
# "r" before the filename means the file path is read literally by python, to avoid conflict with any characters 
# that may have other purposes in python https://discuss.python.org/t/what-does-r-define-in-file-path/3646
# Dataframe to csv file output set to write so it will overwrite anything existing on the csv file each time code is run.
iris.describe()
iris.describe().to_csv(r'iris_summary_descriptive.csv', sep=',', mode='w')

# Summaries seperated by class
# Summary of sepal length
iris[["sepal length in cm", "class"]].groupby("class").describe().to_csv(r'sepal_length_summary.csv', index="class", sep=',', mode='w')

# Summary of sepal width
iris[["sepal width in cm", "class"]].groupby("class").describe().to_csv(r'sepal_width_summary.csv', index="class", sep=',', mode='w')

# Summary of petal length
iris[["petal length in cm", "class"]].groupby("class").describe().to_csv(r'petal_length_summary.csv', index="class", sep=',', mode='w')

# Summary of petal width
iris[["petal width in cm", "class"]].groupby("class").describe().to_csv(r'petal_width_summary.csv', index="class", sep=',', mode='w')

"""
# Histogram of of sepal length
sns.histplot(iris, x = "sepal length in cm", hue="class", kde=True, binwidth=0.2)
plt.grid()
# Saves the plot as a png image
plt.savefig('HistogramSepalLength.png')
plt.clf() # Clears the plot so a new plot can be created. Without this, the subsequent plots are combined into the previous plot

# Histogram of sepal width
sns.histplot(iris, x = "sepal width in cm", hue="class", kde=True, binwidth=0.2,)
plt.grid()
plt.savefig('HistogramSepalWidth.png')
plt.clf()

# Histogram of petal length
sns.histplot(iris, x = "petal length in cm", hue="class", kde=True, binwidth=0.2)
plt.grid()
plt.savefig('HistogramPetalLength.png')
plt.clf()

# Histogram of petal width
sns.histplot(iris, x = "petal width in cm", hue="class", kde=True, binwidth=0.2)
plt.grid()
plt.savefig('HistogramPetalWidth.png')
plt.clf()


# Scatterplots
# Using different markers as well as colour to make the different classes more clear, which is important ahould someone with colourblindness find 
# it tricky to differentiate classes by colour alone.
sns.set_style("whitegrid")
sns.pairplot(iris, hue="class", markers=["o", "s", "D"])
plt.savefig('Pairplot.png')
plt.clf()


# Density function


# Boxplot


# Violin plot
"""

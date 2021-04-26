# An analysis of Fisher’s Iris data set
# Author: Andrew Scott

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading in Fisher’s Iris data set from a .data file
# As file did not include column names, they were added using 'names' as an argument
iris = pd.read_csv('iris.data', names = ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "species"])

# Add check to see if data is complete, no missing values etc 
# https://stackoverflow.com/questions/29530232/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe
print(iris.isnull().values.any()) # Returns false, showing that there are no missing values in this dataset
print(iris.shape) # Prints the number of rows and columns. As expected, we get (150, 5). 
print(iris.info()) # Confirms that we indeed has no null values, row and column size, and shows us the data type in each column

# Overall summary
# Checks for a text file by the specified name, and if it does not exist that file is created. It is in write mode meaning anything written to the 
# file will be deleted and replaced with the output of this code.
with open("iris_summary.txt", "w") as f:
    # Creates a pandas dataframe containing basic descriptive statistics from the dataset. This is then converted to a string so it can be 
    # written to the text file. Two new lines are also specified to seperate this output from subsequent outputs
    descriptives = iris.describe()
    f.write(descriptives.to_string() + "\n\n") 
    
    sepal_length_summary = iris[["sepal length in cm", "species"]].groupby("species").describe()
    f.write(sepal_length_summary.to_string() + "\n\n")

    sepal_width_summary = iris[["sepal width in cm", "species"]].groupby("species").describe()
    f.write(sepal_width_summary.to_string() + "\n\n")

    petal_length_summary = iris[["petal length in cm", "species"]].groupby("species").describe()
    f.write(petal_length_summary.to_string() + "\n\n")

    petal_width_summary = iris[["petal width in cm", "species"]].groupby("species").describe()
    f.write(petal_width_summary.to_string() + "\n\n")


# Also decided to output the summaries to csv files as these can be easier to work with should someone want to do further analysis using 
# the output summaries
# Info to add dataframe to csv file found at: https://stackoverflow.com/questions/31247198/python-pandas-write-content-of-dataframe-into-text-file
# Seperator set to comma so csv file data can easily be added to an excel sheet if needed 
# "r" before the filename means the file path is read literally by python, to avoid conflict with any characters 
# that may have other purposes in python https://discuss.python.org/t/what-does-r-define-in-file-path/3646
# Dataframe to csv file output set to write so it will overwrite anything existing on the csv file each time code is run.
iris.describe().to_csv(r'iris_summary_descriptive.csv', sep=',', mode='w')

# Summaries seperated by species
# Summary of sepal length
iris[["sepal length in cm", "species"]].groupby("species").describe().to_csv(r'sepal_length_summary.csv', index="species", sep=',', mode='w')

# Summary of sepal width
iris[["sepal width in cm", "species"]].groupby("species").describe().to_csv(r'sepal_width_summary.csv', index="species", sep=',', mode='w')

# Summary of petal length
iris[["petal length in cm", "species"]].groupby("species").describe().to_csv(r'petal_length_summary.csv', index="species", sep=',', mode='w')

# Summary of petal width
iris[["petal width in cm", "species"]].groupby("species").describe().to_csv(r'petal_width_summary.csv', index="species", sep=',', mode='w')


# Histogram of of sepal length
sns.histplot(iris, x = "sepal length in cm", hue="species", kde=True, binwidth=0.2)
plt.grid()
# Saves the plot as a png image
plt.savefig('HistogramSepalLength.png')
plt.clf() # Clears the plot so a new plot can be created. Without this, the subsequent plots are combined into the previous plot

# Histogram of sepal width
sns.histplot(iris, x = "sepal width in cm", hue="species", kde=True, binwidth=0.2,)
plt.grid()
plt.savefig('HistogramSepalWidth.png')
plt.clf()

# Histogram of petal length
sns.histplot(iris, x = "petal length in cm", hue="species", kde=True, binwidth=0.2)
plt.grid()
plt.savefig('HistogramPetalLength.png')
plt.clf()

# Histogram of petal width
sns.histplot(iris, x = "petal width in cm", hue="species", kde=True, binwidth=0.2)
plt.grid()
plt.savefig('HistogramPetalWidth.png')
plt.clf()


# Boxplot
# Melt is used to transform the dataframe such that the variables are condensed into one column, which will allow me to call 
# them as an axis on a cat plot: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html 
# This will allow me to show the boxplot for all 4 variables side by side, with the variable
# as the y axis. Idea for this found in a response by user Ian Thompson:
# https://stackoverflow.com/questions/52472757/creating-a-boxplot-facetgrid-in-seaborn-for-python
iris_melt = iris.melt(id_vars='species')

sns.set_style("whitegrid")
sns.catplot(data=iris_melt, x="species", y="value", col="variable", kind = "box", col_wrap=2)
plt.savefig('Boxplot.png')
plt.clf()

# Violin plot
# Violin grids by default had y axis ticks set at 2, so assigning a variable and then adding a line of code to adjust the y axis was used
# as mentioned in: https://cduvallet.github.io/posts/2018/11/facetgrid-ylabel-access 
sns.set_style("whitegrid")
violin = sns.catplot(data=iris_melt, x="species", y="value", col="variable", kind = "violin", col_wrap=2)
violin.set(yticks=list(range(9)))
plt.savefig('Violin.png')
plt.clf()

# Scatterplots
# Using different markers as well as colour to make the different specieses more clear, which is important ahould someone with colourblindness find 
# it tricky to differentiate specieses by colour alone.
sns.set_style("whitegrid")
pair = sns.pairplot(iris, hue="species", markers=["o", "s", "D"])
pair.map_upper(sns.kdeplot, levels=3)
plt.savefig('Pairplot.png')
plt.clf()

# Correlations
correlations = iris.corr()
with open("iris_summary.txt", "a") as f:
    f.write("\t\t"+"  Variable Correlations"+"\n") # Add a title to the output table. Tabs and spaces added to format titles similar to other tables
    f.write(correlations.to_string() + "\n\n")

corr_map = sns.heatmap(correlations, cmap="RdGy", annot=True)
# fixed an initial issue where axis labels were being cut off: https://stackoverflow.com/questions/33660420/seaborn-ticklabels-are-being-truncated
corr_map.figure.tight_layout() 
plt.savefig('Correlation Heatmap.png')
plt.clf()

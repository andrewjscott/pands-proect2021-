# <b>Analysis of the *Iris* Dataset Using Python</b>
# Andrew Scott - Student ID: G00398249


This code was written in [Python 3.8.3](https://www.python.org/downloads/release/python-383/) using [Visual Studio Code version 1.55.2](https://code.visualstudio.com/).   
Python packages not part of the Python Standard Library that were installed and used are:   

[numpy](https://numpy.org/)==1.18.5   
[matplotlib](https://matplotlib.org/)==3.2.2   
[pandas](https://pandas.pydata.org/)==1.0.5   
[seaborn](seaborn.pydata.org/)==0.11.1   

These can be installed by downloading the requirements.txt file and running pip3 install -r requirements.txt. The requirements.txt file was generated using pipreqs 0.4.10.

For this code to work, please ensure that the analysis.py file and the iris.data file are downloaded and stored in the same folder.

# Table of Contents
* <b id="toc1">[Background](#h1)</b>
* <b id="toc2">[Loading the *Iris* Dataset](#h2)</b>
* <b id="toc3">[*Iris* Dataset Summary Statistics](#h3)</b>
    * <b id="toc4">[Overall Summary Table](#h4)</b>
    * <b id="toc5">[Summary Statistics by Variable](#h5)</b>
        * <b id="toc6">[Sepal Length in cm](#h6)</b>
        * <b id="toc7">[Sepal Width in cm](#h7)</b>
        * <b id="toc8">[Petal Length in cm](#h8)</b>
        * <b id="toc9">[Petal Width in cm](#h9)</b>
    * <b id="toc10">[Comma Separated Value Files](#h10)</b>
* <b id="toc11">[Histograms](#h11)</b>
    * <b id="toc12">[Sepal Length Histogram](#h12)</b>
    * <b id="toc13">[Sepal Width Histogram](#h13)</b>
    * <b id="toc14">[Petal Length Histogram](#h14)</b>
    * <b id="toc15">[Petal Width Histogram](#h15)</b>
* <b id="toc16">[Boxplots](#h16)</b>
    * <b id="toc17">[Boxplots of the *Iris* Dataset](#h17)</b>
* <b id="toc18">[Violin plot](#h18)</b>
    * <b id="toc19">[Violin Plots of the *Iris* Dataset](#h19)</b>
* <b id="toc20">[Scatterplots/Pairplot](#h20)</b>
    * <b id="toc21">[Pairplot of the *Iris* Dataset](#h21)</b>
* <b id="toc22">[Correlations](#h22)</b>
    * <b id="toc23">[Variable Correlations Table](#h23)</b>
    * <b id="toc24">[Heatmap](#h24)</b>
        * <b id="toc25">[Correlation Heatmap of the *Iris* Dataset](#25)</b>
* <b id="toc26">[Conclusion](#h26)</b>
* <b id="toc27">[Acknowledgements](#h27)</b>
* <b id="toc28">[References](#h28)</b>

# <b id="h1">Background</b>
The *Iris* dataset first appeared publically in a 1936 paper titled *The use of multiple measurements in taxonomic problems*, by statistician R. A. Fisher<sup id="a1">[1](#f1)</sup>. Collected by botanist Edgar Anderson, who allowed Fisher to analyse and publish the data, the dataset contains 50 samples of three different species of *Iris* flowers - *Iris setosa*, *Iris versicolor*, and *Iris virginica*. Anderson measured four features from each sample - sepal length, sepal width, petal length, and petal width.

![alt text](https://raw.githubusercontent.com/andrewjscott/PandsWork/main/1%20Hh53mOF4Xy4eORjLilKOwA.png "Iris setosa, Iris versicolor, and Iris virginica, with petals and sepals labelled")
Image source<sup id="a2">[2](#f2)</sup> 

Anderson hoped to gain insight into the evolutionary variances among *Iris* flowers that led to their divergence into different species<sup id="a3">[3](#f3)</sup>. For the dataset published by Fisher, all the samples of *Iris setosa* and *Iris versicolor* measured by Anderson grew in the meadows of the Gaspé peninsula in Canada and were measured on the same day, while the samples of *Iris virginica* came from a different colony. Fisher highlighted this difference to make the reader aware that differences in *Iris virginica* from *Iris setosa* and *Iris virginica* may be in part affected by differences in environment. 

In this paper, Fisher used the *Iris* dataset to demonstrate a statistical method that became known as Fisher's linear discriminant. This is a classification method that attempts to project multidimensional data onto a plane that would allow one to more easily identify features that differ between sample points. This allows for the data to be grouped into separate classes based on the features identified as most different between groups after the transformation. Fisher’s linear discriminant is used in a generalised form called linear discriminant analysis (LDA) in machine learning<sup id="a4">[4](#f4)</sup>. 

## Before and after LDA
![alt text](https://raw.githubusercontent.com/andrewjscott/PandsWork/main/Before-LDA-and-after-LDA_.jpg "Before and after LDA")   
Image source<sup id="a5">[5](#f5)</sup> 

# <b id="h2">Loading the *Iris* Dataset</b>
The *Iris* dataset is commonly used as an introductory dataset for both data analysis and machine learning<sup id="a6">[6](#f6)</sup>. This code will be concerned with exploratory data analysis, to gain an understanding of the three *Iris* species measured and what notable patterns can be uncovered through the use of Python. The dataset was downloaded from a UCI repository<sup id="a7">[7](#f7)</sup>.

The following libraries are imported to aid with the investigation of this dataset. By calling them as abbreviated names, this means that when we use any functions that these libraries offer, we can call them by simply typing the abbreviation rather than the library’s full name.

```
import numpy as np
```
NumPy is a library for working with arrays<sup id="a8">[8](#f8)</sup>. NumPy arrays are similar to Python lists, except they are stored in one continuous memory location, which allows for faster and simpler mathematical manipulation. Called as np.  

```
import pandas as pd
```

Pandas is a library built on top of NumPy for working with DataFrames, used for the analysis and manipulation of datasets<sup id="a9">[9](#f9)</sup>. Called as pd.   

```
import matplotlib.pyplot as plt
```

Matplotlib is a plotting library, which allows for the visualisation of data<sup id="a10">[10](#f10)</sup>. Called as plt.   

```
import seaborn as sns
```
Seaborn is a library that offers additional plotting capabilities, built on top of matplotlib<sup id="a11">[11](#f11)</sup>. Called as sns.

The dataset is imported to Python using the following code:   
```
iris = pd.read_csv('iris.data', names = ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "species"])
```   

After learning that functions for .csv files would also work with this .data file<sup id="a12">[12](#f12)</sup>, the pandas function read_csv() is used to import the dataset and convert it into a pandas DataFrame, with the ‘iris.data’ file called as an argument to achieve this. As iris.data is stored in the same folder as the Python code, the filename alone is sufficient. If the file resides in a different folder, the entire path to that file would need to be used instead. By default, the columns have no names, so names were assigned to each column using the names parameter to make it easier to read what each column of data represents. The names were found on UCI in a file called iris.names, which was hosted in the same location as the iris.data file. The iris.name file calls the last column class. However, I changed this column's name to species, as calling it class could be confusing. Taxonomically speaking this column does not contain the class - rather, it contains the species<sup id="a13">[13](#f13)</sup>. The DataFrame is assigned the variable name iris, which means this variable name can be used for any subsequent code that looks at this DataFrame. References to this DataFrame and Python variables containing the word iris will be written with a lower case 'i', whereas reference to the flower *Iris* will be written in italics with a capital 'I'.  

# <b id="h3">*Iris* Dataset Summaries</b>
The first step is to make sure that the dataset is complete and balanced and to get an idea about how the data is structured and laid out. This is to ensure that there are no missing values that may skew the findings. To do this, the following code is used to print out some relevant details about the DataFrame to the terminal. 

```
print(iris.isnull().values.any())
```

This outputs if there are any missing values in the dataset<sup id="a14">[14](#f14)</sup>. If there are, it outputs True and we can then investigate further where this missing value lies. However, in this instance, the output is False, which tells us that there are no null values in this dataset.

```
print(iris.shape)
```

The shape function informs us how many rows and columns the DataFrame has<sup id="a15">[15](#f15)</sup>. As expected, it returns (150, 5). 150 rows for each flower measured, and five columns - four for the measurement variables and one for the flower species.

```
print(iris.info())
```

The info() function confirms that we indeed have no null values, row and column size, and shows us the data type in each column<sup id="a16">[16](#f16)</sup>.

```
print(iris.head())
```
We can also get a look at the DataFrame itself so we can see some of its content and structure. Instead of printing out all 150 rows, we can instead look at just the first five rows using the head() function<sup id="a17">[17](#f17)</sup>. 

```
with open("iris_summary.txt", "w") as f:
```

This code checks if there is a file with the name iris_summary.txt, and if there isn’t, it gets created<sup id="a18">[18](#f18)</sup>. Any summary statistics that are generated will be saved to this file. The “w” indicates the output will be in write mode, which means the output of the following code within the indented block will overwrite anything currently contained in the file. The indentation indicates that the code belongs to the same block. The block generates five sets of summary statistics - one that gives us an overall summary of the entire dataset, and then four summaries of each measured variable for each of the three *Iris* species. The file iris_summary.txt is given the variable name f, which will be used later when writing the code output to the file. 

The next step involves analysing the DataFrame to derive some helpful summary statistics of the entire dataset.

```
descriptives = iris.describe()
    f.write(descriptives.to_string() + "\n\n")
```

First, the pandas function describe() is used to generate a table of summary statistics<sup id="a19">[19](#f19)</sup>. f.write is used to write the summary table generated to the file iris_summary.txt. The argument descriptives.to_string() is used as the table is a DataFrame object, and as such needs to be converted to a string before using the write() function<sup id="a20">[20](#f20)</sup> <sup id="a21">[21](#f21)</sup>. “\n\n” is used to insert some blank lines after the table in the output file. This is done to separate this table from any subsequent output that gets written to the file. 

Looking at the output table, count confirms that we have 150 measurements for each variable. Mean gives the average value, which is the sum of all the values in a column, divided by the number of values. This will give us a good idea of what the typical size of a variable is. However, caution should be taken as an outlier, a value that is considerably larger or smaller than the other values can affect the mean. Std is the standard deviation, which tells us how far from the mean most values lie within. A higher std tells us that the measurements tend to be quite variable in size - as with petal length, whereas a small std tells us that most measurements are similar to each other - see sepal width. Min is the minimum i.e. the smallest measurement in the DataFrame. The table then tells us the values of the measurements that lie on the 25%, 50%, and 75% percentiles. This is helpful to better understand the range of values that are typical for each variable. The 50% value can also be used in conjunction with the mean - if these values differ greatly, this may indicate there’s an outlier that may affect our analysis. The difference between the mean and 50% for the petal length suggests there may be an issue with outliers for this measurement<sup id="a22">[22](#f22)</sup>. 

## <b id="h4">Overall Summary Statistics</b>
|       | sepal length in cm | sepal width in cm | petal length in cm | petal width in cm |
|-------|--------------------|-------------------|--------------------|-------------------|
| count | 150.000000         | 150.000000        | 150.000000         | 150.000000        |
| mean  | 5.843333           | 3.054000          | 3.758667           | 1.198667          |
| std   | 0.828066           | 0.433594          | 1.764420           | 0.763161          |
| min   | 4.300000           | 2.000000          | 1.000000           | 0.100000          |
| 25%   | 5.100000           | 2.800000          | 1.600000           | 0.300000          |
| 50%   | 5.800000           | 3.000000          | 4.350000           | 1.300000          |
| 75%   | 6.400000           | 3.300000          | 5.100000           | 1.800000          |
| max   | 7.900000           | 4.400000          | 6.900000           | 2.500000          |

## <b id="h5">Summary Statistics by Variable</b>
Next, to get a more detailed look at the dataset, code was written to generate summary statistics for each variable, separated by the species. This should provide information as to where the species are most similar and most different from each other. 

```
    sepal_length_summary = iris[["sepal length in cm", "species"]].groupby("species").describe()
    f.write(sepal_length_summary.to_string() + "\n\n")
```

This code is similar to the code for overall statistics. However, the parameters of "sepal length in cm" and "species" are used to determine the variables of interest for the calculated statistics, while groupby(“species”) determines how the output table will be formatted<sup id="a23">[23](#f23)</sup>. The code is repeated, with the variable name changing, to get the following four summary tables.

## <b id="h6">Sepal Length in cm</b>
|                 | count | mean  | std      | min | 25%   | 50% | 75% | max |
|-----------------|-------|-------|----------|-----|-------|-----|-----|-----|
| species         |       |       |          |     |       |     |     |     |
| *Iris-setosa*     | 50.0  | 5.006 | 0.352490 | 4.3 | 4.800 | 5.0 | 5.2 | 5.8 |
| *Iris-versicolor* | 50.0  | 5.936 | 0.516171 | 4.9 | 5.600 | 5.9 | 6.3 | 7.0 |
| *Iris-virginica*  | 50.0  | 6.588 | 0.635880 | 4.9 | 6.225 | 6.5 | 6.9 | 7.9 |

## <b id="h7">Sepal Width in cm</b>
|                 | count | mean  | std      | min | 25%   | 50% | 75%   | max |
|-----------------|-------|-------|----------|-----|-------|-----|-------|-----|
| species         |       |       |          |     |       |     |       |     |
| *Iris-setosa*     | 50.0  | 3.418 | 0.381024 | 2.3 | 3.125 | 3.4 | 3.675 | 4.4 |
| *Iris-versicolor* | 50.0  | 2.770 | 0.313798 | 2.0 | 2.525 | 2.8 | 3.000 | 3.4 |
| *Iris-virginica*  | 50.0  | 2.974 | 0.322497 | 2.2 | 2.800 | 3.0 | 3.175 | 3.8 |

## <b id="h8">Petal Length in cm</b>
|                 | count | mean  | std      | min | 25% | 50%  | 75%   | max |
|-----------------|-------|-------|----------|-----|-----|------|-------|-----|
| species         |       |       |          |     |     |      |       |     |
| *Iris-setosa*     | 50.0  | 1.464 | 0.173511 | 1.0 | 1.4 | 1.50 | 1.575 | 1.9 |
| *Iris-versicolor* | 50.0  | 4.260 | 0.469911 | 3.0 | 4.0 | 4.35 | 4.600 | 5.1 |
| *Iris-virginica*  | 50.0  | 5.552 | 0.551895 | 4.5 | 5.1 | 5.55 | 5.875 | 6.9 |

## <b id="h9">Petal Width in cm</b>
|                 | count | mean  | std      | min | 25% | 50% | 75% | max |
|-----------------|-------|-------|----------|-----|-----|-----|-----|-----|
| species         |       |       |          |     |     |     |     |     |
| *Iris-setosa*     | 50.0  | 0.244 | 0.107210 | 0.1 | 0.2 | 0.2 | 0.3 | 0.6 |
| *Iris-versicolor* | 50.0  | 1.326 | 0.197753 | 1.0 | 1.2 | 1.3 | 1.5 | 1.8 |
| *Iris-virginica*  | 50.0  | 2.026 | 0.274650 | 1.4 | 1.8 | 2.0 | 2.3 | 2.5 |

From the tables, we can see that there is a lot of overlap in sepal length and width between all three species. Of note is that *Iris setosa* typically has a smaller sepal length, than *Iris versicolor* and *Iris virginica*, but despite this *Iris setosa* has wider sepals than the other two species. The biggest difference in species can be seen in the petal length, with *Iris setosa* having much shorter petals than *Iris versicolor*, with *Iris virginica* having the longest petals. In the overall table, we saw the standard deviation for petal length was relatively high, but now when the samples are separated by species, the std is smaller for each. This tells us again that there is a large difference between the species types, but when compared to another sample of the same species, the difference is much smaller. The pattern is similar for petal width, with *Iris setosa* being the narrowest and *Iris virginica* being the widest, but this difference is not as large as it is for petal length.

## <b id="h10">Comma-Separated Value Files</b>
Code has also been written that will generate these tables as comma-separated value(CSV) files. CSV files are not as easy to read by default as the DataFrame tables generated above, but they are easier to use should someone wish to load these tables and do further work and analysis using them<sup id="a24">[24](#f24)</sup>.
```
iris.describe().to_csv(r'iris_summary_descriptive.csv', sep=',', mode='w')
```
This is an example of the code used to generate the CSV files<sup id="a25">[25](#f25)</sup>. The summary table is created using describe(), and the function to_csv() is used to create the CSV file. The name of the output CSV file is placed as an argument, while the "r" before the filename means the file path is read literally by Python, to avoid conflict with any characters that may have other purposes in Python<sup id="a26">[26](#f26)</sup>. The argument sep tells the code to use a comma to separate each value, and mode is set to “w” which stands for write, so the output will write over anything existing on the file. 


# <b id="h11">Histograms</b>
Histograms are used to represent the frequency of a particular variable in a dataset. Code has been written to generate four histograms - one for each of the four measured variables. 

```
sns.histplot(iris, x = "sepal width in cm", hue="species", kde=True, binwidth=0.2,)
plt.grid()
plt.savefig('HistogramSepalWidth.png')
plt.clf()
```

The plots are created using the seaborn function histplot<sup id="a27">[27](#f27)</sup>. The DataFrame iris is passed in as an argument so that that plot will be created using this data. The variable being plotted is passed using x, which plots this variable on the x-axis. This is the only piece of code that changes between the code used to create the four histograms, to generate a different plot for each variable. By default, the y-axis is count, which tells us how many datapoints lie within a particular bin. Specifying the hue as species means that each species will be given a separate colour on the plot, making it possible to differentiate them. KDE being set to True includes a smoothed curved representation of the data, which can be a helpful aid alongside the bin sizes for determining the distribution of the data. Setting binwidth to 0.2 means that each bar on the histogram represents 0.2 cm of data - e.g. the first bar on the Sepal Length plot includes all the data that measured between 4.4 - 4.5 cm, and the following bar is 4.5 - 4.7 cm etc. If the bin size is too large, the plot is unhelpful as too many data points fall into the same bin, whereas if the bin size is too small the histogram can become overly cluttered making it harder to read. 0.2 was chosen as it’s like a happy medium between those extremes for this dataset. 

plt.grid() is a matplotlib function that adds a grid to the plot, which makes it easier to read the values of each bin<sup id="a28">[28](#f28)</sup>. plt.savefig() saves the resulting plot as an image with the filename that has been specified as an argument, which in this example is 'HistogramSepalWidth.png'<sup id="a29">[29](#f29)</sup>. Once a plot has been saved, it is cleared from memory using plt.clf()<sup id="a30">[30](#f30)</sup>. Without this, the existing plot and any subsequent plots will be combined into one plot. This would leave the histogram overly cluttered and too difficult to read.

## <b id="h12">Sepal Length Histogram</b>
![alt text](https://raw.githubusercontent.com/andrewjscott/pands-project2021/main/HistogramSepalLength.png "Sepal Length Histogram")

## <b id="h13">Sepal Width Histogram</b>
![alt text](https://raw.githubusercontent.com/andrewjscott/pands-project2021/main/HistogramSepalWidth.png "Sepal Width Histogram")

## <b id="h14">Petal Length Histogram</b>
![alt text](https://raw.githubusercontent.com/andrewjscott/pands-project2021/main/HistogramPetalLength.png "Petal Length Histogram")

## <b id="h15">Petal Width Histogram</b>
![alt text](https://raw.githubusercontent.com/andrewjscott/pands-project2021/main/HistogramPetalWidth.png "Petal Width Histogram")

While the summary tables when separated by species gave us useful information, plotting this information graphically can tell us about these findings faster than looking through a table of numbers. Many of the conclusions mentioned above after looking at the summary tables can be seen at a glance using histograms. For example, we can see how the petals of *Iris setosa* are all smaller than *Iris versicolor* and *Iris virginica*, and how there is a lot of overlap between all three species when looking at their sepal width. If we were shown three *Iris* flowers at random that were measured for this dataset, one from each species, and tasked with determining what species each flower was, we could be 100% confident in identifying the *Iris setosa* flower by its petal size alone. It would also be reasonable to assume the flower with the largest petals is an *Iris virginica*, but we would be less confident in this answer due to the slight overlap between the petal sizes of *Iris versicolor* and *Iris virginica* among some of the measured samples.
 
The summary tables and histograms complement each other well - the histogram to quickly see patterns in the data, and the tables for precise measurements of the summaries.

# <b id="h16">Boxplots</b>
Another useful plot type for analysing data is the boxplot<sup id="a31">[31](#f31)</sup>. It is useful for showing the quartile ranges of the data, while also indicating if the data is skewed and if there are any outliers.

## How to Interprete a Boxplot</b>
![alt text](https://raw.githubusercontent.com/andrewjscott/PandsWork/main/1%202c21SkzJMf3frPXPAR_gZA.png "How to interprete a Boxplot")
Image source<sup id="a31">[31](#f31)</sup>

As shown in the image above, the box itself shows the range of values that lie between the 25% and 75% range, with the line within the box indicating the median. The lines extending outwards from the box are called whiskers, which show the rest of the data except for the outliers, indicated by a symbol beyond the whiskers. As mentioned earlier, outliers are data points that deviate so far from the norm that they may negatively influence the rest of the analysis.

The following code is used to create boxplots.
```
iris_melt = iris.melt(id_vars='species')

sns.set_style("whitegrid")
sns.catplot(data=iris_melt, x="species", y="value", col="variable", kind = "box", col_wrap=2)
plt.savefig('Boxplot.png')
plt.clf()
```
While the four histogram plots above were generated separately, it is also possible to generate multiple plots on one image<sup id="a32">[32](#f32)</sup>. This is achieved by first transforming the DataFrame so that all the variables are listed in the same column using the melt function<sup id="a33">[33](#f33)</sup>. The resulting DataFrame, given the variable name iris_melt, has 600 rows, and three columns for species, variable (what was measured e.g. sepal length), and value (the measurement e.g. 5.1). The seaborn function set_style() allows us to choose the plot aesthetics, with “whitegrid” giving us a clear and simple plot with a white background and horizontal grid lines<sup id="a34">[34](#f34)</sup>. 

Another seaborn function, catplot(), allows us to generate multiple plots on one image<sup id="a35">[35](#f35)</sup>. To customise the output the following arguments were used - The DataFrame used is the iris_melt DataFrame. The “species” is shown along the x-axis by assigning it to x, while the new column generated by the iris_melt transformation of “value” is assigned to y for it to be used as the y-axis. This would have been much harder to achieve had we used the original iris DataFrame where the values are in different columns. By assigning “variable” to col, we ensure that each variable gets a separate plot. Many kinds of plots can be generated using the catplot() function, and in this instance to make a boxplot, we specify the kind as “box”. Finally, col_wrap set to 2 means that two plots will be generated side by side before moving to a new row. Since we have two plots, this means we get a neat output of two plots on two rows. 

## <b id="h17">Boxplots of the *Iris* Dataset</b>
![alt text](https://raw.githubusercontent.com/andrewjscott/pands-project2021/main/Boxplot.png "Boxplots of the Iris dataset")

From these boxplots, we can again see how the petal sizes for *Iris setosa* are smaller and have a narrower range of values than *Iris versicolor* and *Iris virginica*, who share some overlap. We can also see several outliers among all variables, most notably the petal values for *Iris setosa*, while the sepal length of one *Iris virginica* is significantly smaller than the other recorded values for that species.

# <b id="h18">Violin plot</b>
A violin plot is a plot type that combines the information from both the boxplot and the distribution curves of the histogram<sup id="a36">[36](#f36)</sup>. The black box in the centre of each violin is the information represented by the boxplots, with the thinner lines being the boxplot whiskers. The distribution curve is then plotted alongside the boxplot, and the curve is mirrored on the opposite side which leads to the violin-esque shape. Violin plots won't tell us anything we haven't already learned from the plots we already have, but they are a useful alternative for times when space is limited and can be generated using the following code.

```
sns.set_style("whitegrid")
violin = sns.catplot(data=iris_melt, x="species", y="value", col="variable", kind = "violin", col_wrap=2)
violin.set(yticks=list(range(9)))
plt.savefig('Violin.png')
plt.clf()
```
The code is almost identical to the code used to create the boxplots, with the only difference being that the kind parameter is changed from “box” to “violin”. By default, this resulted in a plot where the y-axis was ticked in intervals of 2, which I felt was too large. The set() function was therefore used to set the y ticks to consist of all integers from 0 to 9<sup id="a37">[37](#f37)</sup>. 

## <b id="h19">Violin Plots of the *Iris* Dataset</b>
![alt text](https://raw.githubusercontent.com/andrewjscott/pands-project2021/main/Violin.png "Violin plots of the Iris dataset")

# <b id="h20">Scatterplots/Pairplot</b>
The summary tables, histograms, box plots, and violin plots are all examples of univariate analysis, which means they look at only one measurement variable at a time. However, it’s also possible to conduct multivariate analysis by looking at the relationship between two variables at a time. Scatterplots allow us to do this, by mapping the point of one variable on the x-axis, and a second variable on the y-axis. As we have four variables, we would require six scatterplots to plot all combinations of variables. By using pairplots, we can quickly generate all combinations of scatterplots on one image<sup id="a38">[38](#f38)</sup>. In addition, A histogram or density curve can be included at the diagonal point where a particular variable is labelled on both the x and y-axis at the same time. Above the diagonal histogram/curves plots we get six more scatterplots, which provide the same information as the plots below the diagonal histogram/curves plots, just with the axes reversed.

```
sns.set_style("whitegrid")
pair = sns.pairplot(iris, hue="species", markers=["o", "s", "D"])
pair.map_upper(sns.kdeplot, levels=3)
plt.savefig('Pairplot.png')
plt.clf()
```
A pairplot can be created using the seaborn function pairplot(). Arguments are passed in for it to use the iris dataset, for the colour to change based on “species” using the parameter hue, and to have different markers for each species, with “o” = circles, “s” = squares, and “D” = diamonds. These were chosen to aid in viewing the points for each species quickly. Different symbols can be helpful for colourblind viewers who may have difficulty differentiating points based on colour alone. 

As the scatterplots above the curve plots provide the same information as the scatterplots below by default, a line of code was added to alter the upper scatterplots. By using the seaborn function map_upper()<sup id="a39">[39](#f39)</sup> and passing in sns.kdeplot as an argument, the upper plots are altered to include circles that group particular clusters of datapoints<sup id="a40">[40](#f40)</sup>. Setting labels to 3 means there are up to three circles drawn for each species. I felt these circles compliment the scatterplots below by highlighting isolated clusters based on size, as well as any overlap between species.

## <b id="h21">Pairplot of the *Iris* Dataset</b>
![alt text](https://raw.githubusercontent.com/andrewjscott/pands-project2021/main/Pairplot.png "Pairplot of the Iris Dataset")

From this plot, we can see how pronounced the difference in petal size is between *Iris setosa*, *Iris versicolor*, and *Iris virginica*. All the *Iris setosa* plants are grouped in a tight cluster with a petal length of less than 2cm and a petal width of less than 0.8cm. We can also see how almost all *Iris versicolor* can be isolated from *Iris virginica* based on petal length and width. There is a small amount of overlap, but in general *Iris virginica* have the largest petal length and width, with *Iris versicolor* falling in the middle. A similar, but not as pronounced pattern can also be observed by looking at petal length along with sepal width. 

# <b id="h22">Correlations</b>
The strength of a relationship between two variables can be shown by calculating their correlation<sup id="a41">[41](#f41)</sup>. Correlations are values between -1 and 1, which calculate the relationship between one variable and another. A positive correlation of 1 indicates that as variable 1 increases, variable 2 also increases at the same rate. A negative correlation of -1 is the inverse, as variable 1 increases, variable 2 decreases at the same rate. While it’s important to be aware of spurious correlations<sup id="a42">[42](#f42)</sup>, they can nevertheless be indicative of an informative relationship between two variables.

```
correlations = iris.corr()
with open("iris_summary.txt", "a") as f:
    f.write("\t\t"+"  Variable Correlations"+"\n") 
    f.write(correlations.to_string() + "\n\n")
```
This code generates a table that shows us the correlation between all variables, and is similar to the code used to generate descriptive tables earlier, except this time using the pandas function corr()<sup id="a43">[43](#f43)</sup>. As with the earlier tables, our text file is called to write the output to the file. This time, “a” is passed as an argument instead of “w” because we don’t want to overwrite everything in the file. Instead, we can append this table to the end of the file through the use of “a”. As the table did not output with a title by default, I added a line of code that writes “Variable Correlations” as a title, with “\t\t” included to tab the heading so it is aligned with the headings for the earlier tables. 

```
correlations.to_csv(r'iris_correlations.csv', sep=',', mode='w')
```
A commas separated values file of the table is also created with this code using the to_csv() function, as was done for the summary tables.

## <b id="h23">Variable Correlations Table</b>
|                    | sepal length in cm | sepal width in cm | petal length in cm | petal width in cm |
|--------------------|--------------------|-------------------|--------------------|-------------------|
| sepal length in cm | 1.000000           | -0.109369         | 0.871754           | 0.817954          |
| sepal width in cm  | -0.109369          | 1.000000          | -0.420516          | -0.356544         |
| petal length in cm | 0.87175            | -0.420516         | 1.000000           | 0.96275           |
| petal width in cm  | 0.817954           | -0.356544         | 0.962757           | 1.000000          |

As we saw in the scatterplot, the relationship between petal length and petal width is strongly positive at 0.96. This means that if we were only shown the petal length of a sample, we could infer the width of that petal with a high degree of accuracy, and vice versa. 

## <b id="h24">Heatmap</b>
While this table is useful, the same information can be plotted on a heatmap to visualise the strength of correlations<sup id="a44">[44](#f44)</sup>. The layout of a heatmap is similar to a pairplot, in that it shows every combination of variables, with the information above the diagonal being a mirror of the information below. 

```
corr_map = sns.heatmap(correlations, cmap="RdGy", annot=True)
corr_map.figure.tight_layout()
plt.savefig('Correlation Heatmap.png')
plt.clf()
```

A heatmap is created using the seaborn function heatmap()<sup id="a45">[45](#f45)</sup>. The correlation table generated is passed in as the data for our heatmap to use. Heatmaps are assigned a colour spectrum to indicate the strength of the correlations for each combination of variables. In this instance, a spectrum of red and grey was chosen by assigning “RdGy” to cmap<sup id="a46">[46](#f46)</sup>. This means that a darker shade of red indicates a stronger negative correlation, while a darker shade of grey shows a stronger positive correlation. By setting annot to True, each segment will also contain the numerical value for that correlation. The resulting heatmap had an issue with the axis labels being cut off. This was fixed by using the function figure.tight_layout()<sup id="a47">[47](#f47)</sup>.

## <b id="h25">Correlation Heatmap of the *Iris* Dataset</b>
![alt text](https://raw.githubusercontent.com/andrewjscott/pands-project2021/main/Correlation%20Heatmap.png "Correlation Heatmap of the Iris dataset")

# <b id="h26">Conclusion</b>
This analysis has highlighted some notable patterns in the *Iris* dataset. First, all the *Iris setosa* flowers measured have considerably smaller petals than *Iris versicolor* and *Iris virginica*, and while the latter two species share some overlap, most *Iris versicolor* are smaller than *Iris virginica*. There is also little variance in the size of *Iris setosa* plants, with all samples having similar dimensions to each other, whereas the sizes of the other two species have greater variance. There is also a strong positive correlation between the petal length and petal width. 

However, there are some limitations of this dataset that must be kept in mind. There are approximately 280 documented species of *Iris*<sup id="a48">[48](#f48)</sup>. We would need further data to determine if the relationships in this dataset hold among other species of *Iris*. One of the strengths of this dataset is also a weakness - namely the fact that all *Iris setosa* and *Iris versicolor* samples were measured on the same day from the same meadow. The strength of this is it limits the effect confounding variables might have on the flower dimensions, such as a different climate and environment. The downside of this is we don’t know how well these measurements generalise to samples of *Iris setosa* and *Iris versicolor* found elsewhere. Indeed, Edgar Anderson wrote that “in Alaska the species itself, *Iris setosa*, is apparently quite as variable as our other American irises”, indicating that the separation in size noted between *Iris setosa* and the other species might not be as pronounced as this analysis implies<sup id="a49">[49](#f49)</sup>. There are also the outliers that were identified in the box plot. While they were left in for this analysis, further analysis could be taken to determine if these measurements negatively influence our findings in any way<sup id="a50">[50](#f50)</sup>. It is also possible to create machine learning code in Python that can learn to categorise these data by species based on their variable measurements<sup id="a51">[51](#f51)</sup>. This includes the LDA method that is based on the analysis conducted by Fisher in his original 1936 paper<sup id="a52">[52](#f52)</sup>. However, code for this is not included in this analysis as I don’t feel my understanding of the code required for machine learning is high enough to justify its inclusion. 

As demonstrated here, Python can be used to create both informative statistical summaries as well as visualisations that aid analysis and increase understanding of a dataset. These outputs can also be exported into documents such as text, CSV, or jpeg files, which provides the ability to share findings with those with no Python knowledge.

# <b id="h27">Acknowledgements</b>
This section will highlight and give credit to sources that were not directly referenced, but still had an influence during the creation of this analysis.

1. The lecture series Programming and Scripting delivered by Andrew Beatty provided the foundational Python knowledge necessary for tackling this dataset.   
2. [A Youtube series delivered by Applied Ai Course](https://www.youtube.com/playlist?list=PLupD_xFct8mFDeCqoUAWZpUddeqmT28_L) was useful for gaining insight as to how to approach analysing the *Iris* dataset.  
3. [A markdown cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) by Adam Pritchard was used to help format the readme.    
4. [A website for generating tables](https://www.tablesgenerator.com/markdown_tables) was used for creating markdown-friendly tables.  
5. The answer by [user Matteo to the question “How to add footnotes to GitHub-flavoured Markdown?”](https://stackoverflow.com/questions/25579868/how-to-add-footnotes-to-github-flavoured-markdown) on Stack Overflow was used to create footnotes that link from the main text to the relevant entry in the reference section, and vice versa. This solution was also adapted to provide links from the Table of Content to the relevant heading section.    
6. [An article detailing some of the history of the *Iris* dataset](https://towardsdatascience.com/the-iris-dataset-a-little-bit-of-history-and-biology-fb4812f5a7b5) by Yong Cui helped provide an overview and historical context, as well as interesting botanical information such as the *Iris* seeds being even more informative about the species than sepal and petal size.     
7. Much of this readme was first written in [Google Docs](docs.google.com/) before being copy-pasted into a markdown file in Visual Studio Code.   
8. The [Grammarly plugin for Firefox](https://www.grammarly.com/) was used to highlight any potential spelling/grammar errors.
 
# <b id="h28">References</b>

<b id="f1">1.</b> Fisher, R.A., 1936. The use of multiple measurements in taxonomic problems. *Annals of eugenics*, 7(2), pp.179-188.[↩](#a1)   
<b id="f2">2.</b> Sporer, Z., 2020. Iris Species Classification — Machine Learning Model. [online] Available at: <https://morioh.com/p/eafb28ccf4e3> [Accessed 27 April 2021].[↩](#a2)   
<b id="f3">3.</b> Anderson, E., 1936. The species problem in Iris. *Annals of the Missouri Botanical Garden*, 23(3), pp.457-509.[↩](#a3)   
<b id="f4">4.</b> McLachlan, G.J., 2004. *Discriminant analysis and statistical pattern recognition* (Vol. 544). John Wiley & Sons, pp.8-9.[↩](#a4)   
<b id="f5">5.</b> Tyagi, N., Introduction to Linear Discriminant Analysis in Supervised Learning. [online] Available at: <https://www.analyticssteps.com/blogs/introduction-linear-discriminant-analysis-supervised-learning> [Accessed 27 April 2021].[↩](#a5)   
<b id="f6">6.</b> Grus, J., 2019. *Data science from scratch: first principles with python.* O'Reilly Media. p.161. [↩](#a6)   
<b id="f7">7.</b> Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. [online] Available at: <http://archive.ics.uci.edu/ml> Irvine, CA: University of California, School of Information and Computer Science. [Accessed 27 April 2021].[↩](#a7)   
<b id="f8">8.</b> Harris, C.R., Millman, K.J., van der Walt, S.J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N.J. and Kern, R., 2020. Array programming with NumPy. Nature, 585(7825), pp.357-362.[↩](#a8)   
<b id="f9">9.</b> The Pandas Development Team, Pandas 1.0.5. Available at: <https://zenodo.org/record/3898987#.YIhUeKEo9PY> [Accessed 27 April 2021].[↩](#a9)   
<b id="f10">10.</b> Hunter, J.D., 2007. Matplotlib: A 2D graphics environment. Computing In Science Engineering 9, 3. 90-95. <https://zenodo.org/record/3898017#.YIhV_6Eo9PY> [Accessed 27 April 2021].[↩](#a10)   
<b id="f11">11.</b> Waskom, M.L., 2021. Seaborn: statistical data visualization. Journal of Open Source Software, 6(60), p.3021.[↩](#a11)   
<b id="f12">12.</b> Adithya, D., 2021. How to read .data files in Python?. [online]. Available at:<https://www.askpython.com/python/examples/read-data-files-in-python> [Accessed 29 April 2021].[↩](#a12)    
<b id="f13">13.</b> ITIS., 2021.  Iris setosa TSN 43195. [online]. Available at: <https://www.itis.gov/servlet/SingleRpt/SingleRpt?search_topic=TSN&search_value=43195#null> [Accessed 27 April 2021].[↩](#a13)   
<b id="f14">14.</b> Anand, S., 2015. How to check if any value is NaN in a Pandas DataFrame. [online]. Available at: <https://stackoverflow.com/questions/29530232/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe> [Accessed 27 April 2021].[↩](#a14)   
<b id="f15">15.</b> Pandas, 2021. pandas.DataFrame.shape. [online]. Available at: <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shape.html> [Accessed 27 April 2021].[↩](#a15)   
<b id="f16">16.</b> Pandas, 2021. pandas.DataFrame.<span></span>info. [online]. Available at: <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.info.html> [Accessed 27 April 2021][↩](#a16)   
<b id="f17">17.</b> Pandas, 2021. pandas.DataFrame.head. [online]. Available at: <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html> [Accessed 29 April 2021].[↩](#a17)   
<b id="f18">18.</b> Ndlovu, V., 2021. Working With Files in Python. [online]. Available at: <https://realpython.com/working-with-files-in-python/> [Accessed 27 April 2021].[↩](#a18)   
<b id="f19">19.</b> Pandas, 2021. pandas.DataFrame.describe. [online]. Available at: <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html> [Accessed 27 April 2021].[↩](#a19)   
<b id="f20">20.</b> Pandas, 2021. pandas.DataFrame.to_string. [online]. Available at: <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_string.html> [Accessed April 27 2021].[↩](#a20)    
<b id="f21">21.</b> Duchamp, R., 2016. Python, Pandas : write content of DataFrame into text File. [online]. Available at: <https://stackoverflow.com/questions/31247198/python-pandas-write-content-of-dataframe-into-text-file> [Accessed 29 April 2021].[↩](#a21)    
<b id="f22">22.</b> Grus, J., 2019. *Data science from scratch: first principles with python.* O'Reilly Media. pp. 61-62.[↩](#a22)   
<b id="f23">23.</b> Pandas, 2021. pandas.DataFrame.groupby. [online]. Available at: <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html> [Accessed 27 April 2021].[↩](#a23)   
<b id="f24">24.</b> Fincher, J., 2018. Reading and Writing CSV Files in Python. [online]. Available at: <https://realpython.com/python-csv/> [Accessed 27 April 2021].[↩](#a24)   
<b id="f25">25.</b> Pandas, 2021. pandas.DataFrame.to_csv. [online]. Available at: <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html> [Accessed 27 April 2021].[↩](#a25)     
<b id="f26">26.</b> Hölscher, M., 2020. What does r’ define in file path. [online]. Available at: <https://discuss.python.org/t/what-does-r-define-in-file-path/3646> [Accessed 28 April 2021].[↩](#a26)    
<b id="f27">27.</b> Seaborn, 2021. seaborn.histplot. [online]. Available at: <https://seaborn.pydata.org/generated/seaborn.histplot.html> [Accessed 27 April 2021].[↩](#a27)   
<b id="f28">28.</b> Matplotlib, 2021. Matplotlib.pyplot.grid. [online]. Available at: <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.grid.html> [Accessed 27 April 2021].[↩](#a28)   
<b id="f29">29.</b> Matplotlib, 2021. Matplotlib.pyplot.savefig. [online]. Available at: <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html> [Accessed 27 April 2021].[↩](#a29)   
<b id="f30">30.</b> Matplotlib, 2021. Matplotlib.pyplot.clf. [online]. Available at: <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.clf.html> [Accessed 27 April 2021].[↩](#a30)   
<b id="f31">31.</b> Galarnyk, M., 2018. Understanding Boxplots. [online]. Available at: <https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51> [Accessed 27 April 2021].[↩](#a31)   
<b id="f32">32.</b> Thompson, I., 2020. Creating a boxplot FacetGrid in Seaborn for python. [online]. Available at: <https://stackoverflow.com/questions/52472757/creating-a-boxplot-facetgrid-in-seaborn-for-python> [Accessed 27 April 2021].[↩](#a32)   
<b id="f33">33.</b> Pandas, 2021. Pandas.melt. [online]. Available at: <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html> [Accessed 27 April 2021].[↩](#a33)   
<b id="f34">34.</b> Seaborn, 2021. Seaborn.set_style. [online]. Available at: <https://seaborn.pydata.org/generated/seaborn.set_style.html> [Accessed 27 April 2021].[↩](#a34)   
<b id="f35">35.</b> Seaborn, 2021. Seaborn.catplot. [online]. Available at: <https://seaborn.pydata.org/generated/seaborn.catplot.html> [Accessed 27 April 2021].[↩](#a35)   
<b id="f36">36.</b> Lewinson, E., Violin plots explained. [online]. Available at: <https://towardsdatascience.com/violin-plots-explained-fb1d115e023d> [Accessed 27 April 2021].[↩](#a36)   
<b id="f37">37.</b> Duvallet, C., 2018. Editing right ylabels in seaborn FacetGrid plots. [online]. Available at: <https://cduvallet.github.io/posts/2018/11/facetgrid-ylabel-access> [Accessed 27 April 2021].[↩](#a37)   
<b id="f38">38.</b> Seaborn, 2021. Seaborn.pairplot. [online]. Available at: <https://seaborn.pydata.org/generated/seaborn.pairplot.html> [Accessed 27 April 2021].[↩](#a38)   
<b id="f39">39.</b> Seaborn, 2021. seaborn.PairGrid.map_upper. [online]. Available at: <https://seaborn.pydata.org/generated/seaborn.PairGrid.map_upper.html> [Accessed 27 April 2021].[↩](#a39)   
<b id="f40">40.</b> Seaborn, 2021. seaborn.kdeplot. [online]. Available at: <https://seaborn.pydata.org/generated/seaborn.kdeplot.html> [Accessed 27 April 2021].[↩](#a40)   
<b id="41">41.</b> Grus, J., 2019. *Data science from scratch: first principles with python.* O'Reilly Media. pp. 64-67.[↩](#a41)   
<b id="f42">42.</b> Vigen, T., 2021. Spurious correlations. [online]. Available at: <https://www.tylervigen.com/spurious-correlations> [Accessed 27 April 2021].[↩](#a42)   
<b id="f43">43.</b> Pandas, 2021. pandas.DataFrame.corr. [online]. Available at: <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html> [Accessed 27 April 2021].[↩](#a43)   
<b id="f44">44.</b> Carvalho, T., 2020. Heatmap Basics with Seaborn. [online]. Available at: <https://towardsdatascience.com/heatmap-basics-with-pythons-seaborn-fb92ea280a6c> [Accessed 27 April 2021].[↩](#a44)   
<b id="f45">45.</b> Seaborn, 2021. Seaborn.heatmap. [online]. Available at: <https://towardsdatascience.com/heatmap-basics-with-pythons-seaborn-fb92ea280a6c> [Accessed 27 April 2021].[↩](#a45)   
<b id="f46">46.</b> Matlibplot, 2021. Choosing Colormaps in Matplotlib. [online]. Available at: <https://matplotlib.org/stable/tutorials/colors/colormaps.html> [Accessed 27 April 2021].[↩](#a46)   
<b id="f47">47.</b> tmdavison, 2015. Seaborn ticklabels are being truncated. [online]. Available at: <https://stackoverflow.com/questions/33660420/seaborn-ticklabels-are-being-truncated> [Accessed 27 April 2021].[↩](#a47)   
<b id="f48">48.</b> United States Department of Agriculture, 2021. Our Native Irises. [online]. Available at: <https://www.fs.fed.us/wildflowers/beauty/iris/index.shtml> [Accessed 27 April 2021].[↩](#a48)   
<b id="f49">49.</b> Anderson, E., 1935. The irises of the Gaspe Peninsula. *Bull. Am. Iris Soc.*, 59, p. 4. [online]. Available at: <https://www.biodiversitylibrary.org/item/270486#page/343/mode/1up> [Accessed 27 April 2021].[↩](#a49)   
<b id="f50">50.</b> Sharma, N., 2018. Ways to Detect and Remove the Outliers [online] <https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba> [Accessed 28 April 2021].[↩](#a50)   
<b id="f51">51.</b> Müller, A.C. and Guido, S., 2016. *Introduction to machine learning with Python: a guide for data scientists.* O'Reilly Media.[↩](#a51)   
<b id="f52">52.</b> scikit-learn, 2021. Comparison of LDA and PCA 2D projection of Iris dataset. [online]. Available at: <https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html> [Accessed 28 April 2021].[↩](#a52)   


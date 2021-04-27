# pands-project2021
# Andrew Scott - Student ID: G00398249

This code was written in Python 3.8.3 using Visual Studio Code version 1.55.2.   
Python packages not part of the Python Standard Library that were installed and used are:   

numpy==1.18.5   
matplotlib==3.2.2   
pandas==1.0.5   
seaborn==0.11.1   

These can be installed by downloading the requirements.txt file and running pip3 install -r requirements.txt.   
The requirements.txt file was generated using pipreqs 0.4.10.

# Background
The Iris dataset first appeared in a 1936 paper titled "The use of multiple measurements in taxonomic problems", by statistician R. A. Fisher[1]. The data was collected by botanist Edgar Anderson, who allowed Fisher to analyse and publish the data. The dataset contains 50 samples of three different species of Iris flowers - Iris setosa, Iris versicolor, and Iris virginica. Anderson measured four features from each sample - sepal length, sepal width, petal length, and petal width.

![alt text](https://raw.githubusercontent.com/andrewjscott/PandsWork/main/1%20Hh53mOF4Xy4eORjLilKOwA.png "Iris setosa, Iris versicolor, and Iris virginica, with petals and sepals labelled")
Image source[2] 

Anderson hoped to gain insight into the evolutionary variances among Iris flowers that led to their divergence into different species[3]. For the dataset published by Fisher, all the samples of Iris setosa and Iris versicolor measured by Anderson grew in the meadows of the Gaspé peninsula in Canada, and were measured on the same day, while the samples of Iris virginica came from a different colony. This difference was highlighted by Fisher to make the reader aware that differences in Iris virginica from Iris setosa and Iris virginica may be in part affected by differences in environment. 

Fisher used the Iris dataset to demonstrate a statistical method that became known as Fisher's linear discriminant. This is a classification method that attempts to project multidimensional data onto a plane that would allow one to more easily identify features that differ between sample points. This in turn allows for the data to be grouped into separate classes based on the features identified as most different between groups after the transformation. Fisher’s linear discriminant is used in a generalised form called linear discriminant analysis(LDA) in machine learning[4]. 

## Before and after LDA
![alt text](https://raw.githubusercontent.com/andrewjscott/PandsWork/main/Before-LDA-and-after-LDA_.jpg "Before and after LDA")
Image source[5] 

# Loading the Iris Dataset
The Iris dataset is commonly used as an introductory dataset to both data analysis and machine learning[6]. This code will be concerned with exploratory data analysis. The dataset was downloaded from a UCI repository[7].

The following libraries are imported to aid with the investigation of this dataset. By calling them as abbreviated names, this means that when we use any methods that these libraries offer, we can call them by simply typing the abbreviation rather than the library’s full name.

```
import numpy as np
```
Numpy is a library for working with arrays[8]. NumPy arrays are similar to python lists, except they are stored in one continuous memory location, which allows for faster and simpler mathematical manipulation. Called as np.  

```
import pandas as pd
```

Pandas is library built on top of NumPy for working with dataframes, used for the analysis and manipulation of datasets[9]. Called as pd.   

```
import matplotlib.pyplot as plt
```

Matplotlib is a plotting library, which allows for the visualisation of data[10]. Called as plt.   

```
import seaborn as sns
```
Seaborn is a library that offers additional plotting capabilities, built on top of matplotlib[11]. Called as sns.

The dataset is imported to python using the following code:   
```
iris = pd.read_csv('iris.data', names = ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "species"])
```   

The pandas method read_csv is used to import the dataset and convert it into a pandas dataframe, with the ‘iris.data’ file called as an argument to achieve this. As iris.data is stored in the same folder as the python code, the filename alone is sufficient. If the file resides in a different folder, the entire path to that file would need to be used instead. By default the columns have no names, so names were assigned to each column using the names parameter to make it easier to read what each column of data represents. The names were found on UCI in a file called iris.names, which was hosted in the same location as the iris.data file. I decided to change the name of the final column as calling it class could be confusing, as taxonomically speaking this column does not contain the class. Rather, it contains the species[12]. The dataframe is assigned the variable name iris, which means this variable name can be used for any subsequent code that looks at this dataframe. 

# Iris Dataset Summaries
The first step is to make sure that the dataset is complete and balanced. This is to ensure that there are no missing values that may skew the findings. 

```
print(iris.isnull().values.any())
```

This outputs if there are any missing values in the dataset[13]. If there are, it outputs True and we can then investigate further as to where this missing value lies. However, in this instance, the output is False, which tells us that there are no null values in this dataset.

```
print(iris.shape)
```

The shape method informs us how many rows and columns the dataframe has[14]. As expected, it returns (150, 5). 150 rows for each flower measured, and 5 columns - 4 for the measurement variables and 1 for the flower species.

```
print(iris.info())
```

The info module confirms that we indeed have no null values, row and column size, and shows us the data type in each column[15].

```
with open("iris_summary.txt", "w") as f:
```

This code checks if there is a file with the name iris_summary.txt, and if there isn’t, it gets created[16]. This file will be used to save any summary statistics that are generated. The “w” indicates the output will be in write mode, which means the output of the following code within the indented block will overwrite anything currently contained in the file. The indentation indicates that the code belongs to the same block. The block generates 5 sets of summary statistics - 1 that gives us an overall summary of the entire dataset, and then 4 summaries of each measured variable for each of the 3 Iris species. The file iris_summary.txt is given the variable name f, which will be used later when writing the code output to the file. 

```
descriptives = iris.describe()
    f.write(descriptives.to_string() + "\n\n")
```


First, the pandas method describe() is used to generate a table of summary statistics[17]. f.write is used to write the summary table generated to the file iris_summary.txt[16]. The argument descriptives.to_string() is needed as the table is a dataframe object, and as such needs to be converted to a string for the write() method to work[18]. “\n\n” is used to insert some blank lines after the table in the output file. This is done to separate this table from any subsequent output that gets written to the file. 

Looking at the output table, count confirms that we have 150 measurements for each variable. Mean gives the average value, which is the sum of all the values in a column, divided by the number of values. This will give us a good idea of what the typical size of a variable is. However, caution should be taken as the mean can be affected by an outlier, a value that is considerably larger or smaller than the other values. Std is the standard deviation, which tells us how far from the mean most values lie within. A higher std tells us that the measurements tend to be quite variable in size - as with petal length, whereas a small std tells us that most measurements are similar to each other - see sepal width. Min is the minimum i.e. the smallest measurement in the dataframe. The table then tells us the values of the measurements that lie on the 25%, 50%, and 75% percentiles. This is helpful to better understand the range of values that are typical for each variable. The 50% value can also be used in conjunction with the mean - if these values differ greatly, this may indicate there’s an outlier that may affect our analysis. The difference between the mean and 50% for the petal length suggests there may be an issue with outliers for this measurement[19]. 

## Overall Summary Table
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

## Summary statistics by Variable
Next, in order to get a more detailed look at the dataset, code was written to generate summary statistics for each variable, separated by the species. This should provide information as to where the species are most similar and most different from each other. 

```
    sepal_length_summary = iris[["sepal length in cm", "species"]].groupby("species").describe()
    f.write(sepal_length_summary.to_string() + "\n\n")
```

This code is similar to the code for overall statistics. However, the parameters of "sepal length in cm" and "species" are used to determine the variables of interest for the calculated statistics, while groupby(“species”) determines how the output table will be formatted[20]. The code is repeated, with the variable name changing, to get the following 4 summary tables.

## Sepal length in cm
|                 | count | mean  | std      | min | 25%   | 50% | 75% | max |
|-----------------|-------|-------|----------|-----|-------|-----|-----|-----|
| species         |       |       |          |     |       |     |     |     |
| Iris-setosa     | 50.0  | 5.006 | 0.352490 | 4.3 | 4.800 | 5.0 | 5.2 | 5.8 |
| Iris-versicolor | 50.0  | 5.936 | 0.516171 | 4.9 | 5.600 | 5.9 | 6.3 | 7.0 |
| Iris-virginica  | 50.0  | 6.588 | 0.635880 | 4.9 | 6.225 | 6.5 | 6.9 | 7.9 |

## Sepal width in cm
|                 | count | mean  | std      | min | 25%   | 50% | 75%   | max |
|-----------------|-------|-------|----------|-----|-------|-----|-------|-----|
| species         |       |       |          |     |       |     |       |     |
| Iris-setosa     | 50.0  | 3.418 | 0.381024 | 2.3 | 3.125 | 3.4 | 3.675 | 4.4 |
| Iris-versicolor | 50.0  | 2.770 | 0.313798 | 2.0 | 2.525 | 2.8 | 3.000 | 3.4 |
| Iris-virginica  | 50.0  | 2.974 | 0.322497 | 2.2 | 2.800 | 3.0 | 3.175 | 3.8 |

## Petal length in cm
|                 | count | mean  | std      | min | 25% | 50%  | 75%   | max |
|-----------------|-------|-------|----------|-----|-----|------|-------|-----|
| species         |       |       |          |     |     |      |       |     |
| Iris-setosa     | 50.0  | 1.464 | 0.173511 | 1.0 | 1.4 | 1.50 | 1.575 | 1.9 |
| Iris-versicolor | 50.0  | 4.260 | 0.469911 | 3.0 | 4.0 | 4.35 | 4.600 | 5.1 |
| Iris-virginica  | 50.0  | 5.552 | 0.551895 | 4.5 | 5.1 | 5.55 | 5.875 | 6.9 |

## Petal width in cm
|                 | count | mean  | std      | min | 25% | 50% | 75% | max |
|-----------------|-------|-------|----------|-----|-----|-----|-----|-----|
| species         |       |       |          |     |     |     |     |     |
| Iris-setosa     | 50.0  | 0.244 | 0.107210 | 0.1 | 0.2 | 0.2 | 0.3 | 0.6 |
| Iris-versicolor | 50.0  | 1.326 | 0.197753 | 1.0 | 1.2 | 1.3 | 1.5 | 1.8 |
| Iris-virginica  | 50.0  | 2.026 | 0.274650 | 1.4 | 1.8 | 2.0 | 2.3 | 2.5 |

# Histograms
Histograms are used to represent the frequency of a particular variable in a dataset. Code has been written to generate 4 histograms - 1 for each of the 4 measured variables. 

```
sns.histplot(iris, x = "sepal width in cm", hue="species", kde=True, binwidth=0.2,)
plt.grid()
plt.savefig('HistogramSepalWidth.png')
plt.clf()
```

The plots are created using the seaborn method histplot[23]. iris is passed in as an argument, so that that plot will be created using the data from the iris dataframe. The variable being plotted is passed using x, which plots this variable on the x-axis. This is the only piece code that changes between the code for the 4 histograms to generate a different plot for each variable. By default, the y-axis is count, which tells how how many datapoints lie within a particular bin. Specifying the hue as species means that each species will be given a separate colour on the plot, making it possible to differentiate them. KDE being set to True includes a smoothed curved representation of the data, which can be a helpful aid alongside the bin sizes for determining the distribution of the data. Setting binwidth to 0.2 means that each bar on the histogram represents 0.2 cm of data - e.g. the first bar on the Sepal Length plot includes all the data that measured between 4.4 - 4.5 cm, and the following bar is 4.5 - 4.7 cm etc. If the bin size is too large, the plot is unhelpful as too many data points fall into the same bin, whereas if the bin size is too small the histogram can become overly cluttered making it harder to read. 0.2 was chosen as it’s like a happy medium between those extremes for this dataset. 

plt.grid() is a matplotlib method that adds a grid to the plot, which makes it easier to read the values of each bin[24]. plt.savefig() saves the resulting plot as an image with the filename that has been specified as an argument, which in this example is 'HistogramSepalWidth.png'[25]. Once a plot has been saved, it is cleared from memory using plt.clf()[26]. Without this, the existing plot and any subsequent plots will be combined together into one plot. This would leave the histogram overly cluttered and too difficult to read.

## Sepal Length Histogram
![alt text](https://raw.githubusercontent.com/andrewjscott/pands-project2021/main/HistogramSepalLength.png "Sepal Length Histogram")

## Sepal Width Histogram
![alt text](https://raw.githubusercontent.com/andrewjscott/pands-project2021/main/HistogramSepalWidth.png "Sepal Width Histogram")

## Petal Length Histogram
![alt text](https://raw.githubusercontent.com/andrewjscott/pands-project2021/main/HistogramPetalLength.png "Petal Length Histogram")

## Petal Width Histogram
![alt text](https://raw.githubusercontent.com/andrewjscott/pands-project2021/main/HistogramPetalWidth.png "Petal Width Histogram")

While the summary tables when separated by species gave us useful information, plotting this information graphically can tell us about these findings faster than looking through a table of numbers. Many of the conclusions mentioned above after looking at the summary tables can be seen at a glance using histograms. For example, we can see how the petals of Iris setosa are all smaller than Iris versicolor and Iris virginica, and how there is a lot of overlap between all 3 species when looking at their sepal width. If we were shown 3 Iris flowers at random that were measured for this dataset, 1 from each species, and tasked with determining what species each flower was, we could be 100% confident in identifying the Iris setosa flower by its petal size alone. It would also be reasonable to assume the flower with the largest petals is an Iris virginica, but we would be less confident in this answer due to the slight overlap between the petal sizes of Iris versicolor and Iris virginica among some of the measured samples.
 
The summary tables and histograms compliment each other well - the histogram for quickly spotting patterns in the data, and the tables for precise measurements of the summaries.

# Boxplots
Another useful plot type for analysing data is the boxplot[27]. It is useful for showing the quartile ranges of the data, while also indicating if the data is skewed in any way and if there are any outliers.

## How to interprete a Boxplot
![alt text](https://raw.githubusercontent.com/andrewjscott/PandsWork/main/1%202c21SkzJMf3frPXPAR_gZA.png "How to interprete a Boxplot")
Image source[27]

As shown in the image above, the box itself shows the range of values that lie between the 25% and 75% range, with the line within the box indicating the median. The lines extending outwards from the box are called whiskers, which show the rest of the data except for the outliers, indicated by a symbol beyond the whiskers. As mentioned earlier, outliers are data points that deviate so far from the norm that they may negatively influence the rest of the analysis.

The following code is used to create boxplots.
```
iris_melt = iris.melt(id_vars='species')

sns.set_style("whitegrid")
sns.catplot(data=iris_melt, x="species", y="value", col="variable", kind = "box", col_wrap=2)
plt.savefig('Boxplot.png')
plt.clf()
```
While the four histogram plots above were generated separately, it is also possible to generate multiple plots on one image[28]. This is achieved by first transforming the dataframe so that all the variables are listed in the same column using the melt method[29]. The resulting dataframe, given the variable name iris_melt, has 600 rows, and 3 columns for species, variable (what was measured e.g. sepal length), and value (the measurement e.g. 5.1). The seaborn method set_style allows us to choose the plot aesthetics, with “whitegrid” giving us a clear and simple plot with a white background and horizonta grid lines[30]. 

Another seaborn method, catplot, allows us to generate multiple plots on one image[31]. To customise the output the following arguments were used - The dataframe used is the iris_melt dataframe. The “species” is shown along the x-axis by assigning it to x, while the new column generated by the iris_melt transformation of “value” is assigned to y for it to be used as the y-axis. This would have been much harder to achieve had we used the original iris dataframe where the values are in different columns. By assigning “variable” to col, we ensure that each variable gets its own separate plot. Many kinds of plots can be generated using the catplot method, and in this instance to make a boxplot, we specify the kind as “box”. Finally, col_wrap set to 2 means that 2 plots will be generated side by side before moving to a new row. Since we have two plots, this means we get a neat output of 2 plots on two rows. 

## Boxplots of the Iris Dataset
![alt text](https://raw.githubusercontent.com/andrewjscott/pands-project2021/main/Boxplot.png "Boxplots of the Iris dataset")

From these boxplots we can again see how the petal sizes for Iris setosa are smaller and have a narrower range of values than Iris versicolor and Iris virginica, who share some overlap. We can also see a number of outliers among all variables, most notably the petal values for Iris setosa, while the sepal length of one Iris virginica is significantly smaller than the other recorded values for that species.

# Violin plot
A violin plot is a plot type that combines the information from both the boxplot and the distribution curves of the histogram[32]. The black box in the center of each violin is the information represented by the boxplots, with the thinner lines being the boxplot whiskers. The distribution curve is then plotted alongside the boxplot, and the curve is mirrored on the opposite side which leads to the violin-esque shape. Violin plots won't tell us anything we haven't already learned from the plots we already have, but they are a useful alternative for times when space is limited, and can be generated using the following code.

```
sns.set_style("whitegrid")
violin = sns.catplot(data=iris_melt, x="species", y="value", col="variable", kind = "violin", col_wrap=2)
violin.set(yticks=list(range(9)))
plt.savefig('Violin.png')
plt.clf()
```
The code is almost identical to the code used to create the boxplots, with the only difference being that the kind parameter is changed from “box” to “violin”. By default this resulted in a plot where the y axis was ticked in intervals of 2, which I felt was too large. The set method was therefore used to set the y ticks to consist of all integers in the range of 0 to 9[33]. 

## Violin Plots of the Iris Dataset
![alt text](https://raw.githubusercontent.com/andrewjscott/pands-project2021/main/Violin.png "Violin plots of the Iris dataset")

# Scatterplots/Pairplot
The summary tables, histograms, box plots, and violin plots are all examples of univariate analysis, which means they look at only one measurement variable at a time. However, it’s also possible to conduct multivariate analysis by looking at the relationship between two variables at a time. Scatterplots allow us to do this, by mapping the point of one variable on the x-axis, and a second variable on the y-axis. As we have 4 variables, we would require 6 scatterplots to plot all possible combinations of variables. By using pairplots, we can quickly generate all possible combinations of scatterplots on one image[34]. In addition, A histogram or density curve can be included at the diagonal point where a particular variable is labeled on both the x and y-axis at the same time. Above the diagonal histogram/curves plots we get 6 more scatterplots, which provide the same information as the plots below the diagonal histogram/curves plots, just with the axes reversed.

```
sns.set_style("whitegrid")
pair = sns.pairplot(iris, hue="species", markers=["o", "s", "D"])
pair.map_upper(sns.kdeplot, levels=3)
plt.savefig('Pairplot.png')
plt.clf()
```
A pairplot can be created using the seaborn method pairplot. Arguments are passed in for it to use the iris dataset, for the colour to change based on “species” using the parameter hue, and to have different markers for each species, with “o” = circles, “s” = squares, and “D” = diamonds. These were chosen to aid viewing the points for each species quickly. Different symbols can be helpful for colourblind viewers who may have difficulty differentiating points based on colour alone. 

As the scatterplots above the curve plots provide the same information as the scatterplots below by default, a line of code was added to alter the upper scatterplots. By using the seaborn method map_upper and passing in sns.kdeplot as an argument, the upper plots are altered slightly to include circles that group particular clusters of datapoints[35][36]. Setting labels to three means there are up to 3 circles drawn for each species. I felt these circles compliment the scatterplots below by highlighting isolated clusters based on size, as well as overlap between species.

## Pairplot of the Iris Dataset
![alt text](https://raw.githubusercontent.com/andrewjscott/pands-project2021/main/Pairplot.png "Pairplot of the Iris Dataset")

From this plot, we can see how pronounced the difference in petal size is between Iris setosa, Iris versicolor, and Iris virginica. All the Iris setosa plants are grouped in a tight cluster with a petal length of less than 2cm and a petal width of less than 0.8cm. We can also see how almost all Iris versicolour can be isolated from Iris virginica based on petal length and width. There is a small amount of overlap, but in general Iris virginica have the largest petal length and width, with Iris versicolor falling in the middle. A similar, but not as pronounced pattern can also be observed by looking at petal length along with sepal width. 

# Correlations
The strength of a relationship between two variables can be shown by calculating their correlation[37]. Correlations are values between -1 and 1, which calculate the relationship between one variable and another. A positive correlation of 1 indicates that as variable 1 increases, variable 2 also increases at the same rate. A negative correlation of -1 is the inverse, as variable 1 increases, variable 2 decreases at the same rate. While it’s important to be aware of spurious correlations[38], they can nevertheless be indicative of an informative relationship between two variables.

```
correlations = iris.corr()
with open("iris_summary.txt", "a") as f:
    f.write("\t\t"+"  Variable Correlations"+"\n") 
    f.write(correlations.to_string() + "\n\n")
```
This code generates a table that shows us the correlation between all variables, and is similar to the code used to generate descriptive tables earlier, except this time using the pandas method corr()[39]. As with the earlier tables, our text file is called to write the output to the file. This time, “a” is passed as an argument instead of “w” because we don’t want to overwrite everything in the file. Instead, we can append this table to the end of the file through the use of “a”. As the table did not output with a title by default, I added a line of code that writes “Variable Correlations” as a title, with “\t\t” included to tab the heading so it is aligned with the headings for the earlier tables. 

```
correlations.to_csv(r'iris_correlations.csv', sep=',', mode='w')
```
A commas separated values file of the table is also created with this code using the to_csv method.

## Variable Correlations Table
|                    | sepal length in cm | sepal width in cm | petal length in cm | petal width in cm |
|--------------------|--------------------|-------------------|--------------------|-------------------|
| sepal length in cm | 1.000000           | -0.109369         | 0.871754           | 0.817954          |
| sepal width in cm  | -0.109369          | 1.000000          | -0.420516          | -0.356544         |
| petal length in cm | 0.87175            | -0.420516         | 1.000000           | 0.96275           |
| petal width in cm  | 0.817954           | -0.356544         | 0.962757           | 1.000000          |

As we saw in the scatterplot, the relationship between petal length and petal width is strongly positive at 0.96. This means that if we were only shown the petal length of a sample, we could infer the width of that petal with a high degree of accuracy, and vice versa. 

While this table is useful, the same information can be plotted on a heatmap to visualise the strength of correlations[40]. The layout of a heatmap is similar to a pairplot, in that it shows every possible combination of variables, with the information above the diagonal being a mirror of the information below. 

```
corr_map = sns.heatmap(correlations, cmap="RdGy", annot=True)
corr_map.figure.tight_layout()
plt.savefig('Correlation Heatmap.png')
plt.clf()
```

## Correlation Heatmap of the Iris Dataset
![alt text](https://raw.githubusercontent.com/andrewjscott/pands-project2021/main/Correlation%20Heatmap.png "Correlation Heatmap of the Iris dataset")

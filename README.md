# pands-project2021

This will be messy and unorganised notes that will later be polished after writing more code.

As iris data set was in .data format, I first had to look up whether reading that filetype into python reqquired any different methods than a csv or json file - https://www.askpython.com/python/examples/read-data-files-in-python


# Summary of sepal length histogram
The curve shows that in general, setosa has the smallest sepal length, followed by versicolor, and finally virginica with the largest sepal length. However, there is significant overlap between species. The most notable takeaway is that 20 of the 50 setosa sepal length fall within 4.9mm and 5.1mm. 

# pairplot

I prefer to have grids on my plot to be able to better read where a point on the plot falls on a particular axis, and plt.grid() worked with the histograms. Howver, it didn't seem to work when I tried to use it to add a grid to the pairplot. Instead, I found that a way to at a grid to a pairplot is by using sns.set_style("whitegrid") before making the pairplot.
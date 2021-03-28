# An analysis of Fisher’s Iris data set
# Author: Andrew Scott

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading in Fisher’s Iris data set from a .data file
# As file did not include column names, they were added using 'names' as an argument
df = pd.read_csv('iris.data', names = ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class"])
print(df)
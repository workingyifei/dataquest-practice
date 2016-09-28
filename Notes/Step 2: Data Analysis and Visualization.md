# Data Analysis

## Getting started with NumPy

```python
	import numpy
	numpy.genfromtxt("file.csv", delimiter=",")	# read file.csv into a NumPy array
	vector = numpy.array([1,2,3,4,5])	# arrays can be constructed using array() method, vector is a one dimension matrix
	matrix = numpy.array([[1,2,3], [4,5,6], [7,8,9]])	# matrix is an multi-dimensional array
```

- Each value in a NumPy array has to have the same data type. A full list of NumPy data types can be found [here](http:#docs.scipy.org/doc/numpy-1.10.1/user/basics.types.html).
- NumPy will automatically figure out an appropriate data type when reading in data or converting *lists* to *arrays*. using *dtype* property, the data type can be found.
- [astype()](http:#docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.ndarray.astype.html) method can be used to conver data type of an array to specified
- sum() -- computes the sum of all the elements in a vector, or the sum along a dimension in a matrix
- mean() -- computes the average of all the elements in a vector, or the average along a dimension in a matrix
- max() -- computes the maximum of all the elements in a vector, or the maximum along a dimension in a matrix
- **axis = 1** means that we want to perform the operation on each *row*, and **axis=0** means on each *column*
- Limitations with NumPy:
	- A whole array has to be the same datatype, which makes it cumbersome to work with many datasets.
	- Columns and rows have to be referred to by number, which gets confusing when you go back and forth from column name to column number

## Getting started with Pandas

- To represent tabular data, Pandas uses a custom data structure called a DataFrame. A DataFrame is a highly efficient, 2-dimensional data structure that provides a suite of methods and attributes to quickly explore, analyze, and visualize data. The DataFrame object is similar to the NumPy 2D array but adds support for many features that help you work with tabular data.

- One of the biggest advantages that Pandas has over NumPy is the ability to store mixed data types in rows and columns. Many tabular datasets contain a range of data types and Pandas DataFrames handle mixed data types effortlessly while NumPy doesn't. Pandas DataFrames can also handle missing values gracefully using a custom object, NaN, to represent those values. A common complaint with NumPy is its lack of an object to represent missing values and people end up having to find and replace these values manually. In addition, Pandas DataFrames contain axis labels for both rows and columns and enable you to refer to elements in the DataFrame more intuitively. Since many tabular datasets contain column titles, this means that DataFrames preserve the metadata from the file around the data.

- The Series object is a core data structure that Pandas uses to represent rows and columns. A Series is a labelled collection of values similar to the NumPy vector. The main advantage of Series objects is the ability to utilize non-integer labels. NumPy arrays can only utilize integer labels for indexing.

- In Python, we have the **None** keyword and type, which indicates no value
- The Pandas library uses **NaN**, which stands for "not a number", to indicate a missing value

- In Pandas, dataframes and series have row indices. These work just like column indices, and can be values like numbers, characters, and strings.

- Series objects use NumPy arrays for fast computation, but build on them by adding valuable features for analyzing > data. For example, while NumPy arrays utilize an integer index, Series objects can utilize other index types, like a string index. Series objects also allow for mixed data types and utilize the NaN Python value for handling missing values.

 - Three key data structures in Pandas are:
 	- Series (collection of values)

 	- DataFrame (collection of series objects)
 	- Panel (collectino of DataFrame objects)

- [iloc[] vs. loc[] vs. ix[]](http:#stackoverflow.com/questions/31593201/pandas-iloc-vs-ix-vs-loc-explanation)
	- *iloc* works on the **positions** in the index (so it only takes integers)
	- *loc* works on **labels** in the index
	- *ix* usually tries to behave like *loc* but falls back to behaving like *iloc* if the label is not in the index

```python
	import pandas as pd
	dataframe = pd.read_csv("filename.csv")
	print(type(dataframe))	#this will print out type of filename which is 'pandas.core.frame.DataFrame'
	dataframe.head(5)	# first 5 rows 
	dataframe.columns 	# full list of column names
	dataframe.shape 		# returns a tuple of integers representing number of rows and columns
	dataframe.loc[5] 		# series object representing row with index 5 
	dataframe.loc[[2,5,10]]	# select multiple rows
	dataframe["column_name"].tolist() 	# convert column names into a list
	dataframe.sort_values(by, axis=0, ascending=True, inplace=False) 	# sort by the values along either axis
	pd.isnull(filename["column_name"]) 		# return a series with boolean values
	dataframe["column_name"].mean() 		# column mean
	dataframe["column_name"].sum()			# column sum
	dataframe.pivot_table()					# Create a spreadsheet-style pivot table as a DataFrame
	dataframe.dropna() 						# drop missing values
	dataframe.reset_index()					# reindex, make new indeces starting from 0
	dataframe.apply(func, axis)				# axis=0 apply function on columns, axis=1 apply function on rows
	dataframe["column_name"].unique()		# return the unique values in a column
	dataframe["column_name"].index.tolist() # return index as a list
	dataframe["column_name"].reindex() 		# return a new series with new index
	dataframe.set_index("column_name", inplace, drop)		# pass column name as index, inplace=True will set the index to the current dataframe, drop=False will keep the column specified in the index
	series.sort_index() 					# return a series with sorted index
	seires.sort_values()					# return a series with sorted values

```

# Data Visualization
## Commonly used matplotlib plots

```python
	import matplotlib.pyplot as plt
	plt.scatter(x, y)		# scatter plot
	plt.plot(x, y)			# line plot
	plt.bar(x, y)			# bar plot
	plt.barh(x, y)			# horizontal bar plot
	plt.title("title")		
	plt.xlabel("xlabel")
	plt.ylabel("ylabel")
	plt.styoe.use("ggplot")	# use ggplot (the style of the popular R plotting library) as plot aesthetics
	plt.xticks(rotation=90)	# rotation x label 90 degree
```

## Pandas plots

```python
	df.hist("column_name")	# histogram plot, can use layout and bin to customize it
	df.boxplot(by="column_name")	# generate a box plot
```
## Seaborn
- Seaborn should be thought of as a complement to matplotlib, not a replacement for it. When using seaborn, it is likely that you will often invoke matplotlib functions directly to draw simpler plots already available through the pyplot namespace. Further, while the seaborn functions aim to make plots that are reasonably “production ready” (including extracting semantic information from Pandas objects to add informative labels), full customization of the figures will require a sophisticated understanding of matplotlib objects.

- Pandas vs. Seaborn
 - Plotting using Pandas

	The short answer is that you want to use Pandas to visualize data when you don't need to customize the resulting plot much and Seaborn when you want more extensive control using a simple API. The plotting functions available to Pandas DataFrame objects don't expose many parameters for customization of a plot and are primarily useful for data exploration (which is why the last mission used them heavily!). Customizing a plot generated from a Pandas DataFrame plotting method means working directly with the Matplotlib representation, which requires a much deeper understanding of Matplotlib (which we'll cover in a later lesson in this course).

 - Plotting using Seaborn

	Seaborn, on the other hand, provides a good amount of customization for its plots both through its plotting functions' parameters and through top-level API functions on the Seaborn object. While both Pandas and Seaborn expose the underlying Matplotlib representation for thorough customization, only Seaborn allows us to customize the data visualizations we generate using a simple API instead of having to dive into Matplotlib for common tweaks.

```python
	import matplotlib.pyplot as plt
	import seaborn as sns
	%matplotlib inline

	sns.set_style('dark')
	sns.distplot(births['birthord'], kde=False, axlabel='Birth number')
	sns.plt.show()

	births = pd.read_csv('births.csv')
	sns.boxplot(x='birthord', y='agepreg', data=births)
	sns.plt.show()

	sns.pairplot(births[['agepreg','prglngth','birthord']])
	sns.plt.show()
```

## Under the hood of Matplotlib

```python
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	ax.set(xlim=(0,13), ylim=(10,110),xlabel="Month", ylabel="Temperature", title="Year Round Temperature")
	ax.scatter(month, temperature, color='darkblue', marker='o') #add xlabel, ylabel, and title
	plt.show()
```
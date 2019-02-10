
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns                                                       # the commonly used alias for seaborn is sns

from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

help(pd.DataFrame.loc)
type(variable_name)

sns.set_style("whitegrid")                                                  # set a seaborn style of your taste

#Numpy
#*************************************************

np.ones((5, 3)|[5,3], dtype = np.int)                                       #: Create array of 1s,tuple or list anything ;  dtype by default is float
np.zeros(4, dtype = np.int)                                                 #: create 1d array
np.random.random([3, 4])					                                #: Create array of 3*4 of random numbers
np.arange(10, 100, 5)	| np.arange(24)						                #: Create array with increments of a fixed step size; # From 10 to 100 with a step of 5  ;  from 0 to 23 with step of 1
np.linspace(1, 2, 4)							                            #: Create array of fixed length  ; # Array of length 4 between 1 and 2 excluding (2)
np.arange(24).reshape(2, 3, 4)                                              # Creating a 3-D array of (z*X*y) 2*3*4 ; # reshape() simply reshapes a 1-D array 
np.indices((3,3))                                                           # create 3 d array
some_array.reshape(4, -1)					                                # If you specify -1 as a dimension, the dimensions are automatically calculated : # -1 means "whatever dimension is needed" 

array_1 = np.arange(12).reshape(3, 4)
array_2 = np.arange(20).reshape(5, 4)				
np.vstack((array_1, array_2))					                            # vstack : # Note that np.vstack(a, b) throws an error - you need to pass the arrays as a list
np.hstack()							                                        # horizontal stack


arr = np.ones((2,3))

arr.shape: Shape of array (n x m)
arr.dtype: data type (int, float etc.)
arr.ndim: Number of dimensions (or axes)
arr.itemsize: Memory used by each array elememnt in bytes

print(array_1d[[2, 5, 6, 8]])                                                  # Specific elements ; # Notice that array[2, 5, 6] will throw an error, you need to provide the indices as a list
print(array_1d[2:])                                                          # Slice third element onwards
print(array_1d[0::2])                                        		         # Subset starting 0 at increment of 2 

print(array_2d[1, :])                                        		        #: entire 2nd row
print(array_2d[:, 2])                                        		        #: entire 3rd column
print(array_2d[:, :3])							                            # Slicing all rows and the first three columns

np.sin(a)
np.cos(a)
np.log(a)
np.exp(a)

a_list = [x/(x+1) for x in a]                                               #  normal non numpy way

f = np.vectorize(lambda x: x/(x+1))
f(a)                                                                         #  vectorize numpy way 

houses['Outlier'] = np.where(houses['Bathrooms'] < 20, 0, 1)                # Create feature based on boolean condition



#Pandas
#**************************************************************************

s = pd.Series([2, 4, 5, 6, 9])                                                  # A series is similar to a 1-D numpy array ; # Creating a numeric pandas series
date_series = pd.date_range(start = '11-09-2017', end = '12-12-2017')               # creating a series of type datetime
s[2:]   s[[1, 3]]                                                                   # accessing element from 2nd index till end     # accessing the second and the fourth elements

df = pd.DataFrame({'name': ['Vinay', 'Kushal', 'Aman', 'Saif'], 
                'age': [22, 25, 24, 28], 
                'occupation': ['engineer', 'doctor', 'data analyst', 'teacher']})

market_df = pd.read_csv("market_fact.csv")
market_df.head()
market_df.tail()
market_df.info()                                                            # Looking at the datatypes of each column
market_df.describe()                                                        # Describe gives you a summary of all the numeric columns in the dataset ; like mean ,median etc...

market_df.columns                                                           # Column names
market_df.shape                                                             # dimension ; # The number of rows and columns

df['column'] or df.column                                                   #return a series
df[['col_x', 'col_y']]                                                      #returns a dataframe

market_df.set_index('Ord_id', inplace = True)                               # Setting index to Ord_ida
market_df.sort_index(axis = 0, ascending = False)                           # Sorting by index  # axis = 0 indicates that you want to sort rows (use axis=1 for columns)
market_df.sort_values(by='Sales',ascending = False).head()                  # Sorting by values # Sorting in increasing order of Sales
market_df.sort_values(by=['Prod_id', 'Sales'], ascending = False)           # Sorting by more than two columns ; # Sorting in prod_id and then look for  sales

# df.iloc and df.loc
#********************************

df.loc(row,col)                                                                            # syntax  for displaying data frame
market_df.iloc[[3, 7, 8]]                                                       # Select multiple rows using a list of indices
market_df.iloc[[True, True, False, True, True, False, True]]                    # Using booleans    # This selects the rows corresponding to True
market_df.loc[4:8,]                                                            # Selecting rows using a range of labels # Notice that with df.loc, both 4 and 8 are included, unlike with df.iloc   # This is an important difference between iloc and loc
market_df.loc['Ord_5406', ['Sales', 'Profit', 'Cust_id']]
market_df.loc[['Ord_5406', 'Ord_5446', 'Ord_5485'], 'Sales':'Profit']               # Select multiple orders using labels, and some columns

X = np.array([[1.1, 11.1], 
              [2.2, 22.2], 
              [3.3, 33.3], 
              [4.4, 44.4], 
              [np.nan, 55]])
df = pd.DataFrame(X, columns=['Col1', 'Col2'])                                      #  load data as data frame

#Subsetting dataframe Rows Based on Conditions
#*******************************************

df.loc[df['Sales'] > 3000]  | df.loc[df['Sales'] > 3000, :]                     # selecting rows based on boolean condition
df.loc[(df.Sales > 2000) & (df.Sales < 3000) & (df.Profit > 100), :]            # E.g. all orders having 2000 < Sales < 3000 and Profit > 100  ; # Also, this time, you need all the columns
df.loc[(df.Sales > 2000) & (df.Sales < 3000) , ['Cust_id', 'Sales', 'Profit']]       # E.g. all orders having 2000 < Sales < 3000 and Profit > 100  ; # Also, this time, you only need the Cust_id, Sales and Profit columns

customers_in_bangalore = ['Cust_1798', 'Cust_1519', 'Cust_637', 'Cust_851']
df.loc[df['Cust_id'].isin(customers_in_bangalore), :]                             # syntax  df['column_name'].isin(list)

# Select rows from a dataframe  |  Select columns from a dataframe  |   Select subsets of dataframes
#*************************************************
market_df[5::3].head()                                                      # selecting rows from 5 till end with  a step of 3 
sales = market_df['Sales']                                                  #  for seleting single column
salesnew = market_df[['Sales','Discount']]                                             #  for seleting multiple column ; pass as an list
market_df.set_index('Ord_id').head()                                        # making index as one of the colum field

#Subsetting Data in pandas by Data Type
#****************************************************************
beverage['beer_servings'] = beverage.beer_servings.astype(float)                    # to change type of column  .astype() 
beverage = pd.read_csv(url, dtype={'beer_servings':float})                          # change data type of column while loading
beverage.beer_servings_type.astype('category')

beverage_Numeric = beverage.select_dtypes(include=['int64']).copy()                 # use df.select_dtypes to select the columns by data type

#Merge multiple dataframes using common columns/keys using pd.merge()  |   Concatenate (pile on top)(if column names r not  same) dataframes using pd.concat()
************************************************************
#Note that you can also use the pd.concat() method to merge dataframes using common keys, though here we will not discuss that. For simplicity, we have used the pd.merge() method for database-style merging and pd.concat() for appending dataframes having no common columns.
#merge -  use generally if column names r same   |||| and   concat -  use generally if column names r not same


pd.concat([df1, df2], axis = 0)  | df1.append(df2)                          # concat or append  azis =0  ie row wise stack
pd.concat([df1, df2], axis = 1)                                             # used when different column names r there and we need to concat column side by side in  dataframe


# Splitting data into groups |   Applying a function to each group (e.g. mean or total sales)  |  Combining the results into a data structure showing the summary statistics
#*******************************************************************

df_1 = pd.merge(market_df, customer_df, how='inner', on='Cust_id')            # Merging the dataframes one by one
df_2 = pd.merge(df_1, product_df, how='inner', on='Prod_id')
df_3 = pd.merge(df_2, shipping_df, how='inner', on='Ship_id')

m1 = pd.merge(btc, ether, how="inner", left_on="Date_btc", right_on="Date_et")

df_by_segment = df.groupby('Customer_Segment')
master_df['Customer_Segment'].unique()  | df_by_segment['Profit'].sum()
df_by_segment['Profit'].sort_values(ascending = False)

pd.DataFrame(df_by_segment['Profit'].sum())                                         # converting to dataframe

df['Order_Date'] = pd.to_datetime(df['Order_Date'])                                 #converting object type to datetime format
time_df = df.groupby('Order_Date')['Sales'].sum()                                   #groupby 
df['month'] = df['Order_Date'].dt.month                                             # extract month from date part

by_product_cat_subcat = master_df.groupby(['Product_Category', 'Product_Sub_Category'])         # 1. Group by category and sub-category
by_product_cat_subcat['Profit'].mean()                                                          # then aplly mean
master_df.groupby('Region').Profit.mean()                                                   # E.g. Customers in which geographic region are the least profitable?


#Managing Outliers
#***********************************************************************
df.dropna()                                                                         # Drop observations with missing values in data frame

houses['Outlier'] = np.where(houses['Bathrooms'] < 20, 0, 1)                            # Create feature with value 0(for true) or 1(for false) based on boolean condition

df = pd.DataFrame()                                                                 # Create an empty dataset


df['x0'] = [0.3051,0.4949,0.6974,0.3769,0.2231,0.341,0.4436,0.5897,0.6308,0.5]
df['x1'] = [np.nan,0.2654,0.2615,0.5846,0.4615,0.8308,0.4962,0.3269,0.5346,0.6731]      # Create two variables called x0 and x1. Make the first value of x1 a missing value
# Create an imputer object that looks for 'Nan' values, then replaces them with the mean value of the feature by columns (axis=0)
mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_df = Imputer(strategy='most_frequent', axis=0)

# Train the imputor on the df dataset
mean_imputer = mean_imputer.fit(df)

# Apply the imputer to the df dataset
imputed_df = mean_imputer.transform(df.values)  | imputer_df.fit_transform(X)


Encoding Ordinal Categorical Features               ="http://contrib.scikit-learn.org/categorical-encoding/"
***************************************************************
mapper = {'Low':1, 'Medium':2,'High':3}                                     # Create mapper

Employeedf['Satisfaction'] = Employeedf['Satisfaction'].replace(mapper)         # Map feature values to scale
Employeedf.head()

Encoding Nominal Categorical Features                  =" http://www.handsonmachinelearning.com/blog/McTKK/python-one-hot-encoding-with-scikit-learn"
********************************************************************

Employeedf = pd.get_dummies(Employeedf, columns=['Department'],drop_first=True)
Employeedf = pd.get_dummies(Employeedf, columns=['Gender'],drop_first=True)


min max strategy
*********************************************************
min_max_scaler = preprocessing.MinMaxScaler()                           # Create scale

np_scaled = min_max_scaler.fit_transform(df)                           # Transform the feature

df_normalized = pd.DataFrame(np_scaled,columns=('salary','score'))
df_normalized

standardization rescales
******************************************************

Standardization rescales data to have a mean (μ) of 0 and standard deviation (σ) of 1 (unit variance).

scaler = preprocessing.StandardScaler()                                 # Create scaler

standardized = scaler.fit_transform(df)                                 # Transform the feature

df_standardized = pd.DataFrame(standardized,columns=('salary','score'))             # Show feature

Binning
*******************************************************

bins = [20,30,40,70]
labels = [1,2,3]
df['binned_age'] = pd.cut(df['age'], bins=bins, labels=labels)
print (df)



#Matplot
#********************************************

# Plotting multiple lines on the same plot

x = np.linspace(0, 5, 10)
y = np.linspace(3, 6, 10)

plt.plot(x, y)        |     plt.plot([1,2,3,4],[5,6,7,8])                           # plotting x and y array                       
plt.plot(x, y, 'r-', x, y**2, 'b+', x, y**3, 'g^')                              # plot three curves: y, y**2 and y**3 with different line types
plt.xlabel("Current")
plt.ylabel("Voltage")
plt.title("Ohm's Law")
plt.xlim([20, 80])
plt.ylim([200, 800])
plt.show()

#Figures and Subplots
#************************************************
    matplot = "https://matplotlib.org/users/pyplot_tutorial.html"
    matplotTutorial ="https://github.com/rougier/matplotlib-tutorial"
    seaborn = "https://seaborn.pydata.org/tutorial/categorical.html"
    seaborn heatmap = "https://seaborn.pydata.org/generated/seaborn.heatmap.html"

plt.subplot(nrows, ncols, nsubplot)


x = np.linspace(1, 10, 100)
y = np.log(x)
 
plt.figure(1)                                                                            # Optional command, since matplotlib creates a figure by default anyway # initiate a new figure explicitly

# Create a subplot with 1 row, 2 columns 

                                                                                # create the first subplot in figure 1 
plt.subplot(221)                                                                # equivalent to plt.subplot(1, 2, 1)
plt.title("221")
plt.plot(x, y)

                                                                                # create the second subplot in figure 1
plt.subplot(222)
plt.title("222")                                                                # equivalent to plt.subplot(1, 2, 2)
plt.plot(x, y**2)


plt.subplot(223)                                                                # equivalent to plt.subplot(2, 2, 1)
plt.title("223")
plt.plot(x, y)

                                                                                # create the second subplot in figure 1
plt.subplot(224)
plt.title("224")                                                                 # equivalent to plt.subplot(2, 2, 2)
plt.plot(x, y**2)

plt.show()                                                                              # it will plot as 2* 2 matrix
                                                                                        221,222
                                                                                        223,224



#Boxplots   |   Histograms  |   Scatter plots   | Bar plots
#*********************************************************************
plt.boxplot(df['Order_Quantity'])   | sns.boxplot(df['Order_Quantity'])
plt.show()                                                                          # we have write .show() to see the image
plt.yscale('log')                                                                   # log scale subplot


plt.hist(df['Sales'])                                                               # Histograms
plt.show()

plt.scatter(df['Sales'], df['Profit'])                                              # Scatter plots with two variables: Profit and Sales

image = plt.imread("number.png")                                                        # reading a PNG image
plt.imshow(image)
plt.show()


sns.distplot(df['Shipping_Cost'])                                               # simple density plot
sns.distplot(df['Shipping_Cost'][:200], rug=True)                               # rug = True # plotting only a few points since rug takes a long while
sns.distplot(df['Sales'], hist=False)                                           # Simple density plot (without the histogram bars) can be created by specifying hist=False.

Since seaborn uses matplotlib behind the scenes, the usual matplotlib functions work well with seaborn. For example, you can use subplots to plot multiple univariate distributions.

plt.subplot(2, 2, 1)
plt.title('Sales')
sns.distplot(df['Sales'])

sns.boxplot(y=df['Order_Quantity'])                                                 # to plot the values on the vertical axis, specify y=variable
sns.boxplot(x='Product_Category', y='Sales', data=df)                               # boxplot of a variable across various product categories(group by category  with sales)

#Bivariate                                                                              #two univariate distributions plotted on x and y axes respectively.
sns.jointplot('Sales', 'Profit', df)
sns.jointplot('Sales', 'Profit', df, kind="hex", color="k")

btc.columns = btc.columns.map(lambda x: str(x) + '_btc')                            # putting a suffix(_btc) with column names 
sns.pairplot(curr)                                                                  # pairplot Pairwise Scatter Plot for all variables

cor = curr.corr()                                                                   # You can also observe the correlation between the currencies # using df.corr()
round(cor, 3)

plt.figure(figsize=(10,8))                                                          # figure size
sns.heatmap(cor, cmap="YlGnBu", annot=True)                                         # heatmap helpful to visualise the correlation matrix itself using sns.heatmap()

plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')                 # set figure size for larger figure
sns.boxplot(x='Customer_Segment', y='Profit', hue="Product_Category", data=df)              # specify hue="categorical_variable"

sns.boxplot(x=df['Product_Category'], y=100*df['Shipping_Cost']/df['Sales'])                # different format

sns.barplot(x='Product_Category', y='Sales', data=df)                                       # bar plot with default statistic=mean
sns.barplot(x='Product_Category', y='Sales', data=df, estimator=np.median)                  # subplot 2: statistic=median

sns.countplot(y="Product_Sub_Category", data=df)                                    # Plotting count across a categorical variable

sns.tsplot(df_time)                                                                 # time series plot

year_month = pd.pivot_table(df, values='Sales', index='year', columns='month', aggfunc='mean')  #Pivoting the data using 'month'   index = rows,  mean of sales will be shown at 0,0  0,1  0,2 and so on 

Managing missing values
**********************************************
X[np.isnan(X).any(axis=0)]                                                          # display the rows with nan as any one column value


Doubts

what is the significance of line on bar plot     "C:\Users\p7111567\ML\Session 5 - Pandas and Vizualizations\Basics of visualisation\3 Plotting Categorical and Time-Series Data\3_Plotting_Categorical_Data.ipynb
"

X = np.array([[1.1, 11.1], 
              [2.2, 22.2], 
              [3.3, 33.3], 
              [4.4, 44.4], 
              [np.nan, 55]])
X[np.isnan(X).any(axis=0)]                                                          # display the rows with nan as any one column value




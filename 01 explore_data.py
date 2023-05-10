# required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## prettifies seaborn
sns.set_style('whitegrid')

## if using Jupyter Notebook:
#%matplotlib inline

#################################################

# data cleaning
"""
drop outliers and irrelevant data here
"""

"""
UNDERSAMPLING: balances uneven datasets by keeping all of the data in the minority class and selecting randomly from majority class

column_to_predict = ''

majority = df[df[column_to_predict] == 0]
minority = df[df[column_to_predict] == 1]

# grab random samples (as many as there are in minority class)
undersample = []
for i in minority[column_to_predict].values:
    undersample.append(majority.iloc[random.randint(0, len(majority)-1)])

undersample = pd.DataFrame(undersample)

undersampled_data = pd.concat([minority, undersample])
"""

## read csv file
df = pd.read_csv('file.csv')

# get number of unique values
for column in df:
  print(f"{column}: {df[column].nunique()}")

## one-hot encoding. Look for invalid data. Turn them into numbers
invalid_col = df.select_dtypes(['object']).columns
dummies = pd.get_dummies(df[invalid_col])

"""
drop columns with:
- 2 categories or more
"""
col_to_drop = [""]
dummies.drop(col_to_drop, axis=1, inplace=True)

### add encoded columns. Drop invalid columns from df
df_new = pd.concat([df.drop(invalid_col, axis=1), dummies], axis=1)

# Visualize data
display(df.head())
print('\nDescribe\n')
display(df.describe())
print('\nInformation\n')
display(df.info())

# Exploring data (aka Exploratory Data Analysis)
numeric_types = ['int64', 'float64'] # select only numerical values
numerical_df = df.select_dtypes(include=numeric_types)
sns.heatmap(numerical_df.corr(), annot=True)
sns.pairplot(df) # if dataset is big, comment this line out
sns.histplot(x='column', data=df)
sns.countplot(x='column', data=df)
sns.scatterplot(x='', y='', data=df)

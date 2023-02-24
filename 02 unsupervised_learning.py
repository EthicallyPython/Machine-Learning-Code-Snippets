# required libraries
import pandas as pd

## if using Jupyter Notebook:
#%matplotlib inline

#################################################

# data cleaning
## read csv file
df = pd.read_csv('file.csv')

## split data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# MACHINE LEARNING
## import machine learning library model library. Also sets variable equal to model
from sklearn.family import Model
model = Model()

## training and testing model
model.fit(df)

# If you don't have labels (original answers) for machine, you don't need to evaluate its performance.

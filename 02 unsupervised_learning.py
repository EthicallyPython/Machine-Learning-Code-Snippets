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

"""
# NOT REQUIRED IF YOU DON'T HAVE LABELS (ANSWERS FOR MACHINE)

# EVALUATE MACHINE PERFORMANCE
## seeing how well model did
y_test = test_data
predictions = model.labels_

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

## mean errors. Checks how close estimates are to actual values
from sklearn.metrics import mean_absolute_error, mean_squared_error
print(mean_absolute_error(y_test, predictions))
print(mean_squared_error(y_test, predictions))
### RMSE (root mean squared error)
print(np.sqrt(mean_squared_error(y_test, predictions)))
"""

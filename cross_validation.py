"""
CROSS VALIDATION: split data into training and test sets

below are different types of cross-validation that you can perform on data.
"""

## hold-out
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=155)

## k-fold cross-validation: split into a certain number of groups
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
kf.get_n_splits(X)

X_train, X_test, y_train, y_test = [], [], [], []

### NOTE: you will need to loop over train and test data when preprocessing and training model 
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train.append(X[train_index])
    X_test.append(X[test_index])
    y_train.append(y[train_index])
    y_test.append(y[test_index])

# MACHINE LEARNING. Use `pipeline.py` instead if your ML project requires multiple steps
column_to_predict = ''

## X are your inputs
X = df.drop(column_to_predict, axis=1).values

## y is what you want to predict
y = df[column_to_predict].values

## split data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## import machine learning library model library. Also sets variable equal to model
from sklearn.family import Model
model = Model()

## training and testing model
model.fit(X_train, y_train)
predictions = model.predict(X_test)


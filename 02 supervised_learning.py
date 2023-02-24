# MACHINE LEARNING. Use `pipeline.py` instead if your ML project requires multiple steps
## variables
X = df['numeric_column'] # can use more than one column
y = df['column_to_predict']

## split data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## import machine learning library model library. Also sets variable equal to model
from sklearn.family import Model
model = Model()

## training and testing model
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# EVALUATE MACHINE PERFORMANCE
## classification: seeing how well model did
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
print(classification_report(y_test, predictions))
print(ConfusionMatrixDisplay.from_predictions(y_test, predictions))

### flipping predicted and true for clarity
cm = confusion_matrix(y_test, predictions)
cmp = ConfusionMatrixDisplay(cm, display_labels=['label_1', 'label_2'])

plt.xlabel('Actual')
plt.ylabel('Predicted')

### remove grid lines for confusion matrix
plt.grid(False)
plt.show()

## regression: mean errors. Checks how close estimates are to actual values
from sklearn.metrics import mean_absolute_error, mean_squared_error
print(mean_absolute_error(y_test, predictions))
print(mean_squared_error(y_test, predictions))
### RMSE (root mean squared error)
print(np.sqrt(mean_squared_error(y_test, predictions)))

## show relations between column and number. 
coeff = pd.DataFrame(model.coef_, X.columns, columns=['Coeff'])

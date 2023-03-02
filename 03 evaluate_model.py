# EVALUATE MACHINE PERFORMANCE
## CLASSIFICATION
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

### Deep Learning: uncomment below if using deep learning
#predictions = [np.round(prediction) for prediction in predictions]

print(classification_report(y_test, predictions))
print(ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=[False, True]))



## REGRESSION
from sklearn.metrics import mean_absolute_error, mean_squared_error

print(mean_absolute_error(y_test, predictions))
print(mean_squared_error(y_test, predictions))
### RMSE (root mean squared error)
print(np.sqrt(mean_squared_error(y_test, predictions)))

## show relations between column and number. For logistic and linear regression only
coef = pd.DataFrame(model.coef_, X.columns, columns=['Coeff'])


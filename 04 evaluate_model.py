# EVALUATE MACHINE PERFORMANCE
## CLASSIFICATION
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

print(classification_report(y_test, predictions))
print(ConfusionMatrixDisplay.from_predictions(y_test, predictions))

### flipping predicted and true for clarity
cm = confusion_matrix(y_test, predictions)
cmp = ConfusionMatrixDisplay(cm, display_labels=['label_1', 'label_2'])

### removes lines if you're using seaborn
sns.reset_orig()

## REGRESSION
from sklearn.metrics import mean_absolute_error, mean_squared_error

print(mean_absolute_error(y_test, predictions))
print(mean_squared_error(y_test, predictions))
### RMSE (root mean squared error)
print(np.sqrt(mean_squared_error(y_test, predictions)))

## show relations between column and number. For logistic and linear regression only
coef = pd.DataFrame(model.coef_, X.columns, columns=['Coeff'])

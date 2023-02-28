# column you want to use for predictions
column_to_predict = ''

# X are your inputs
X = df.drop(column_to_predict, axis=1).values

# y is what you want to predict
y = df[column_to_predict].values

## split data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## scale data. ALL DATA MUST BE SCALED
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() #replace with any scaler
  
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#######################################################################################

# MACHINE LEARNING
from tensorflow.keras.models import Sequential # base model
from tensorflow.keras.layers import Dense, Dropout # adds layers

model = Sequential()

## input layer. Based on number of features
model.add(Dense(len(df.columns), activation='relu'))

## hidden layers
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))

# dropout layers. Prevents overfitting
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.5)) # turns off 50% of neurons randomly at this layer

## output layer. Change activation depending on problem type
model.add(Dense(1, activation='relu'))


## compiling. Loss depends on problem (i.e. classification, regression, etc.)
### For a multi-class classification problem
#model.compile(optimizer='adam',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])

### For a binary classification problem
#model.compile(optimizer='adam',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])

### For a mean squared error regression problem
#model.compile(optimizer='adam',
#              loss='mse')

## prevent overfitting
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
							patience=25) # waits 25 epochs before stopping model

## fit model to data
model.fit(x=X_train,
          y=y_train,
          epochs=250,
          validation_data=(X_test, y_test),
          batch_size=128, # smaller = longer, but less likely to overfit
          callbacks = [early_stop]
        )

# EVALUATE MODEL
predictions = model.predict(X_test)
"""
if loss or accuracy diverges, you are overfitting
"""
## loss
loss = pd.DataFrame(model.history.history)

### visualization
loss_train = loss['loss']
loss_val = loss['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

## accuracy
#todo: implement

# round predictions for classification report
predictions = [np.round(prediction) for prediction in predictions]

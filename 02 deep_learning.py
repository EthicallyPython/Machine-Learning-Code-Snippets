## split data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## scale data. ALL DATA MUST BE SCALED
from sklearn.preprocessing import MinMaxScaler()
scaler = MinMaxScaler() #replace with any scaler
  
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# adding layers
from tensorflow.keras.models import Sequential # base model
from tensorflow.keras.layers import Dense, Dropout # adds layers
model = Sequential()

## input layer. Based on number of features
model.add(Dense(4, activation='relu'))

## hidden layers
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))

# dropout layers. Prevents overfitting
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.5)) # turns off 50% of neurons randomly at this layer

## output layer. Change activation depending on problem type
model.add(Dense(1, activation='relu'))


## compiling. Optimizer can change depending on problem
### For a multi-class classification problem
#model.compile(optimizer='rmsprop',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])

### For a binary classification problem
#model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])

### For a mean squared error regression problem
#model.compile(optimizer='rmsprop',
#              loss='mse')


model.fit(x=X_train, y=y_train,
          epochs=250,
          validation_data=(X_test, y_test),
         	batch_size=128 # smaller = longer, but less likely to overfit
      )


loss = pd.DataFrame(model.history.history)

# if loss and val loss are diverging, you're overfitting
loss.plot()

# model metrics
predictions = model.predict(X_test)


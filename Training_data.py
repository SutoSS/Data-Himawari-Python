import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.metrics import accuracy_score

#Load Dataset
data = pd.read_csv("olahdata1.csv")
data = data.astype('float32')
data = data.to_numpy()

#Select Data Training and Testing
training = data[0:130]
testing = data[130:]

#Select Coloum Array
training_features = training[:,1:-1]
training_labels = training[:,0]
testing_features = testing[:,1:-1]
testing_labels = testing[:,0]

#Show shape input
shape_value = training_features[0].shape
print(shape_value)

#Build the model
model = Sequential()
model.add(Input(shape=(5,)))
model.add(Dense(1, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(training_features, training_labels, epochs=50, validation_data = (testing_features,testing_labels))

y_hat = model.predict(testing_features)
print(y_hat)

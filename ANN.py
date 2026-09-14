# ANN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_excel('BookANN.xlsx')

X = pd.DataFrame(dataset.iloc[:, 12:35].values)
y = dataset.iloc[:,36].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.loc[:, 15] = labelencoder_X.fit_transform(X.iloc[:, 15])
labelencoder_X_26 = LabelEncoder()
X.loc[:, 19] = labelencoder_X.fit_transform(X.iloc[:, 19])
labelencoder_X_27 = LabelEncoder()
X.loc[:, 20] = labelencoder_X.fit_transform(X.iloc[:, 20])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units=6, activation = 'relu', input_dim = 23))
classifier.add(Dense(units= 6, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test,y_pred)
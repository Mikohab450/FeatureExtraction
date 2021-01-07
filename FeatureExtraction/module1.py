from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Dense
from keras import models

               

n=69+666

model = Sequential()
model.add(Dense(32, input_dim=4096,activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(n, activation='relu'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
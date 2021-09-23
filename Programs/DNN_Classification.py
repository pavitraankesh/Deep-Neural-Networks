import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

age = [[42],[16],[5],[67],[18],[10],[34],[9],[45],[54],[12],[36]]
eligible = [1,0,0,1,1,0,1,0,1,1,0,1]

model = Sequential()
model.add(Dense(500, activation='relu',input_dim=1))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(age,eligible, epochs=200)

test=[[24]]
print(model.predict(test))
